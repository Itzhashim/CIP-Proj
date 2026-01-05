"""
AI-Powered Intelligent Threat Detection and Response System
Main Orchestration Pipeline

This module only orchestrates the training pipeline by importing
and calling functions from specialized modules.
"""

import os
import sys
import warnings
import numpy as np

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import custom modules
from network_traffic import train_network_classifier
from web_intrusion import train_wids
from malware_analysis import train_malware_detector

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def run_demo_predictions():
    """Run sample predictions to demonstrate system functionality"""
    print("\n" + "="*80)
    print("RUNNING DEMONSTRATION PREDICTIONS")
    print("="*80)
    
    try:
        # Load network traffic model
        print("\n[DEMO] Loading Network Traffic model...")
        network_model_data = joblib.load(os.path.join(MODEL_DIR, 'network_rf.pkl'))
        network_model = network_model_data['model']
        network_scaler = network_model_data['scaler']
        
        # Simulate a test sample
        print("[DEMO] Testing Network Traffic Classification...")
        n_features = network_scaler.n_features_in_
        test_sample = np.random.randn(1, n_features)
        test_sample_scaled = network_scaler.transform(test_sample)
        prediction = network_model.predict(test_sample_scaled)[0]
        
        if prediction == 1:
            print("  ✓ ALERT: Network attack detected!")
        else:
            print("  ✓ Normal network traffic")
        
    except Exception as e:
        print(f"[DEMO] Network model demo error: {str(e)}")
    
    try:
        # Load web intrusion model
        print("\n[DEMO] Loading Web Intrusion Detection model...")
        wids_model_data = joblib.load(os.path.join(MODEL_DIR, 'wids_iforest.pkl'))
        wids_model = wids_model_data['model']
        wids_scaler = wids_model_data['scaler']
        
        print("[DEMO] Testing Web Intrusion Detection...")
        # Simulate anomalous web traffic
        test_web = np.array([[5000, 100, 1000, 150, 0.8, 1]])  # Suspicious pattern
        test_web_scaled = wids_scaler.transform(test_web)
        prediction = wids_model.predict(test_web_scaled)[0]
        
        if prediction == -1:
            print("  ✓ ALERT: Web anomaly detected!")
        else:
            print("  ✓ Normal web traffic")
        
    except Exception as e:
        print(f"[DEMO] WIDS model demo error: {str(e)}")
    
    try:
        # Load malware model
        print("\n[DEMO] Loading Malware Analysis model...")
        malware_model = keras.models.load_model(os.path.join(MODEL_DIR, 'malware_lstm.h5'))
        malware_tokenizer = joblib.load(os.path.join(MODEL_DIR, 'malware_tokenizer.pkl'))
        
        print("[DEMO] Testing Malware Analysis...")
        # Simulate suspicious behavior
        test_behavior = ["dll_inject hook_install keylogger_start privilege_escalate"]
        test_seq = malware_tokenizer.texts_to_sequences(test_behavior)
        test_padded = pad_sequences(test_seq, maxlen=100, padding='post')
        prediction = malware_model.predict(test_padded, verbose=0)[0][0]
        
        if prediction > 0.5:
            print(f"  ✓ ALERT: Malware detected (confidence: {prediction:.2%})")
        else:
            print(f"  ✓ Benign behavior (confidence: {1-prediction:.2%})")
        
    except Exception as e:
        print(f"[DEMO] Malware model demo error: {str(e)}")


def main():
    """Main execution pipeline - orchestrates all modules"""
    
    print("="*80)
    print("AI-POWERED THREAT DETECTION AND RESPONSE SYSTEM")
    print("="*80)
    print()
    
    print("\n" + "="*80)
    print("STARTING TRAINING PIPELINE")
    print("="*80)
    print()
    
    success_count = 0
    total_modules = 3
    results = {}
    
    # MODULE 1: Network Traffic Classification
    print("\n[PIPELINE] Invoking Network Traffic Classifier...")
    try:
        network_results = train_network_classifier()
        if network_results['success']:
            success_count += 1
            results['network'] = network_results
    except Exception as e:
        print(f"\n[ERROR] Module 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # MODULE 2: Web Intrusion Detection
    print("\n[PIPELINE] Invoking Web Intrusion Detection System...")
    try:
        wids_results = train_wids()
        if wids_results['success']:
            success_count += 1
            results['wids'] = wids_results
    except Exception as e:
        print(f"\n[ERROR] Module 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # MODULE 3: Malware Analysis
    print("\n[PIPELINE] Invoking Malware Analysis System...")
    try:
        malware_results = train_malware_detector()
        if malware_results['success']:
            success_count += 1
            results['malware'] = malware_results
    except Exception as e:
        print(f"\n[ERROR] Module 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Run demonstration
    if success_count > 0:
        try:
            run_demo_predictions()
        except Exception as e:
            print(f"\n[ERROR] Demo failed: {str(e)}")
    
    # Final status
    print("\n" + "="*80)
    print("FINAL SYSTEM STATUS")
    print("="*80)
    print()
    
    if success_count == total_modules:
        print("✓ Network attack detection: OPERATIONAL")
        print("✓ Web anomaly detection: OPERATIONAL")
        print("✓ Malware detection: OPERATIONAL")
        print()
        print(f"SUCCESS: All {total_modules} modules trained and ready!")
        print()
        print("Models saved in:", MODEL_DIR)
        print("  - network_rf.pkl")
        print("  - wids_iforest.pkl")
        print("  - malware_lstm.h5")
        print("  - malware_tokenizer.pkl")
        
        # Show module metrics
        if 'network' in results:
            print(f"\nNetwork Classifier Accuracy: {results['network']['accuracy']:.4f}")
        if 'wids' in results:
            print(f"WIDS Anomalies Detected: {results['wids']['anomaly_count']}/{results['wids']['total_samples']}")
        if 'malware' in results:
            print(f"Malware Detector Accuracy: {results['malware']['accuracy']:.4f}")
    else:
        print(f"PARTIAL SUCCESS: {success_count}/{total_modules} modules operational")
        if success_count > 0:
            print("\nSome models are still functional for demonstration purposes.")
    
    print("\n" + "="*80)
    print("SYSTEM READY")
    print("="*80)
    print()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
