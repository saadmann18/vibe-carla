#!/usr/bin/env python3
"""
Test Runner for Week 1 EKF Project
Runs all test suites and generates a comprehensive test report
"""

import subprocess
import sys
import os

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_test_suite(test_file, description):
    """Run a specific test suite"""
    print_header(description)
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "-s",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + "/..")
    return result.returncode == 0

def main():
    """Main test runner"""
    print_header("Week 1 EKF Test Suite")
    
    results = {}
    
    # Run Unit Tests (TC-001, TC-002)
    results['Unit Tests'] = run_test_suite(
        "tests/week1/test_ekf_unit.py",
        "TC-001 & TC-002: Unit Tests (Initialization & Motion Model)"
    )
    
    # Run Integration Tests (TC-003) - Requires CARLA
    print("\n⚠️  Integration tests require CARLA simulator running!")
    print("    Start CARLA: cd /home/saad/dev/carlaSim/CARLA_0.9.15 && ./CarlaUE4.sh")
    print("    Start Driver: python PythonAPI/examples/manual_control.py\n")
    
    response = input("CARLA running? (y/n): ").strip().lower()
    if response == 'y':
        results['Integration Tests'] = run_test_suite(
            "tests/week1/test_ekf_integration.py",
            "TC-003: Integration Tests (End-to-End Fusion)"
        )
        
        # Run Performance Tests (TC-004, TC-005, TC-006)
        results['Performance Tests'] = run_test_suite(
            "tests/week1/test_ekf_performance.py",
            "TC-004, TC-005, TC-006: Performance & Validation Tests"
        )
    else:
        print("⏭️  Skipping CARLA-dependent tests")
        results['Integration Tests'] = None
        results['Performance Tests'] = None
    
    # Print Summary
    print_header("Test Summary")
    for suite, passed in results.items():
        if passed is None:
            status = "⏭️  SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{suite:30s} {status}")
    
    print("\n" + "="*70 + "\n")
    
    # Return exit code
    if all(v is True or v is None for v in results.values()):
        print("🎉 All executed tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
