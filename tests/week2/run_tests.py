#!/usr/bin/env python3
"""
Test Runner for Week 2 Radar-LiDAR Fusion Project
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_test_suite(test_file, description):
    print_header(description)
    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "-s", "--tb=short"]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + "/../..")
    return result.returncode == 0

def main():
    print_header("Week 2 Radar-LiDAR Fusion Test Suite")
    results = {}
    
    # Run Unit Tests
    results['Unit Tests'] = run_test_suite(
        "tests/week2/test_fusion_unit.py",
        "Module Unit Tests (Utils, Tracking, Metrics)"
    )
    
    # Run Integration Tests - Requires CARLA
    print("\n⚠️  Integration tests require CARLA simulator running!")
    response = input("CARLA running? (y/n): ").strip().lower()
    
    if response == 'y':
        results['Integration Tests'] = run_test_suite(
            "tests/week2/test_fusion_integration.py",
            "End-to-End Integration Tests (CARLA Sensor Flow)"
        )
    else:
        print("⏭️  Skipping CARLA-dependent tests")
        results['Integration Tests'] = None
    
    # Print Summary
    print_header("Test Summary")
    for suite, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        if passed is None: status = "⏭️  SKIPPED"
        print(f"{suite:30s} {status}")
    
    return 0 if all(v is True or v is None for v in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
