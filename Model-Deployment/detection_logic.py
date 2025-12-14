import time
from datetime import datetime
import numpy as np
import os

class InstanceDetector:
    """Detect PPE compliance per instance with configurable settings"""
    
    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.last_non_compliant_time = 0
        self.last_activity_time = 0
        self.current_instance_id = None
        self.instance_counter = 0
        self.in_non_compliant_period = False
        self.snapshot_counter = 0
        
        self.non_compliance_delay = 3  # seconds before declaring non-compliant
        self.non_compliance_start_time = None  # when non-compliance was first detected
        self.pending_non_compliant = False  # waiting for delay to pass
        
        self.required_ppe_settings = {
            'helmet': True,
            'vest': True,
            'mask': False
        }
    
    def update_settings(self, settings):
        """Update detection settings"""
        if 'required_ppe' in settings:
            self.required_ppe_settings = settings['required_ppe']
        if 'non_compliance_delay' in settings:
            self.non_compliance_delay = settings['non_compliance_delay']
        if 'instance_reset_timeout' in settings:
            self.window_seconds = settings['instance_reset_timeout']
        print(f"[InstanceDetector] Settings updated: delay={self.non_compliance_delay}s, timeout={self.window_seconds}s, ppe={self.required_ppe_settings}")
    
    def process_detection(self, all_detections, dev_mode=False, settings=None):
        """Process all detections and determine instance"""
        current_time = time.time()
        
        if settings:
            self.update_settings(settings)
        
        has_person = any(d['class'].lower() == 'person' for d in all_detections)
        
        if not has_person:
            self.pending_non_compliant = False
            self.non_compliance_start_time = None
            
            if self.in_non_compliant_period and self.last_activity_time > 0:
                time_since_last = current_time - self.last_activity_time
                if time_since_last > self.window_seconds:
                    print(f"[InstanceDetector] Resetting instance - no person for {time_since_last:.1f}s (timeout: {self.window_seconds}s)")
                    self.in_non_compliant_period = False
                    self.current_instance_id = None
                    self.snapshot_counter = 0
                    self.last_activity_time = 0
            
            return {
                'instance_id': None,
                'has_person': False,
                'is_compliant': True,
                'missing_ppe': [],
                'detected_ppe': [],
                'should_capture': False
            }
        
        self.last_activity_time = current_time
        
        ppe_items = [d for d in all_detections if d['class'].lower() != 'person']
        detected_classes = [item['class'].lower() for item in ppe_items]
        detected_ppe = list(set(detected_classes))
        
        missing_ppe = []
        
        ppe_class_mapping = {
            'helmet': ['helmet', 'hardhat', 'hard-hat', 'hard hat'],
            'vest': ['vest', 'safety-vest', 'safety vest'],
            'mask': ['mask', 'face mask', 'face-mask', 'respirator']
        }
        
        for ppe_type, is_required in self.required_ppe_settings.items():
            if is_required:
                class_names = ppe_class_mapping.get(ppe_type, [ppe_type])
                found = any(cls in detected_classes for cls in class_names)
                if not found:
                    missing_ppe.append(ppe_type)
        
        raw_is_compliant = len(missing_ppe) == 0
        
        if not raw_is_compliant:
            if not self.pending_non_compliant:
                self.pending_non_compliant = True
                self.non_compliance_start_time = current_time
                print(f"[InstanceDetector] Non-compliance detected, starting {self.non_compliance_delay}s delay timer")
            
            time_since_detection = current_time - self.non_compliance_start_time
            if time_since_detection >= self.non_compliance_delay:
                is_compliant = False
            else:
                is_compliant = True
        else:
            self.pending_non_compliant = False
            self.non_compliance_start_time = None
            is_compliant = True
        
        if not is_compliant:
            if not self.in_non_compliant_period:
                self.instance_counter += 1
                self.current_instance_id = f"instance{self.instance_counter}"
                self.in_non_compliant_period = True
                self.snapshot_counter = 0
                print(f"[InstanceDetector] New instance started: {self.current_instance_id}")
            
            self.last_non_compliant_time = current_time
            
            return {
                'instance_id': self.current_instance_id,
                'has_person': True,
                'is_compliant': False,
                'missing_ppe': missing_ppe,
                'detected_ppe': detected_ppe,
                'should_capture': True
            }
        else:
            if self.in_non_compliant_period:
                time_since_last_violation = current_time - self.last_non_compliant_time
                if time_since_last_violation > self.window_seconds:
                    print(f"[InstanceDetector] Ending instance - compliant for {time_since_last_violation:.1f}s (timeout: {self.window_seconds}s)")
                    self.in_non_compliant_period = False
                    self.current_instance_id = None
                    self.snapshot_counter = 0
            
            return {
                'instance_id': None,
                'has_person': True,
                'is_compliant': True,
                'missing_ppe': [],
                'detected_ppe': detected_ppe,
                'should_capture': False
            }
    
    def get_next_snapshot_filename(self):
        """Get next snapshot filename for current instance"""
        if self.current_instance_id:
            self.snapshot_counter += 1
            return f"{self.current_instance_id}_snapshot_{self.snapshot_counter}"
        return None


class ComplianceChecker:
    """Check overall compliance"""
    
    def check_compliance(self, instance_result, dev_mode=False):
        """Return compliance status"""
        if not instance_result['has_person']:
            return True
        
        return instance_result['is_compliant']


class SnapshotManager:
    """Handle snapshot saving"""
    
    def __init__(self, snapshot_dir='snapshots'):
        self.snapshot_dir = snapshot_dir
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
    
    def save_snapshot(self, frame, filename):
        """Save frame as snapshot with specific filename"""
        try:
            import cv2
            
            if not filename:
                return None
            
            filepath = os.path.join(self.snapshot_dir, f"{filename}.jpg")
            
            if frame is not None and frame.size > 0:
                success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success and os.path.exists(filepath):
                    print(f"[v0] Snapshot saved: {filepath}")
                    return filepath
                else:
                    print(f"[v0] Failed to save snapshot: {filepath}")
            
            return None
        except Exception as e:
            print(f"[v0] Error saving snapshot: {e}")
            return None
