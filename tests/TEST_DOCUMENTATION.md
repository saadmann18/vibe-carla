# Week 1 EKF Test Documentation

## Test Suite Overview

This document describes the comprehensive test suite for the Week 1 Extended Kalman Filter (EKF) sensor fusion project, aligned with the DOORS requirements matrix.

## Test Requirements Matrix

| Req ID | Requirement | Category | Priority | Verification Method | Status | Test Cases |
|--------|-------------|----------|----------|---------------------|--------|------------|
| SF-EKF-001 | Initialize 6D state [x,y,vx,vy,yaw,yaw_rate] | Functional | HIGH | Unit Test | ✅ PASS | TC-001.1, TC-001.2 |
| SF-EKF-002 | Predict using constant velocity + turn model | Algorithm | HIGH | Simulation | ✅ PASS | TC-002.1, TC-002.2 |
| SF-EKF-003 | Fuse GPS [x,y] + IMU [yaw_rate] | Integration | CRITICAL | End-to-End | 🔄 RUN TO VERIFY | TC-003.1-003.3 |
| SF-EKF-004 | Fused RMSE < GPS RMSE by ≥30% | Performance | HIGH | Quantitative | 🔄 RUN TO VERIFY | TC-004.1 |
| SF-EKF-005 | Process CARLA data @ 10Hz | Interface | Medium | Stress Test | 🔄 RUN TO VERIFY | TC-005.1 |
| SF-EKF-006 | Position error σ < 2.0m (1000 steps) | Non-Functional | HIGH | Validation | 🔄 RUN TO VERIFY | TC-006.1 |

## Test Cases

### TC-001: Initialization Verification

#### TC-001.1: State Initialization to Zeros
- **Test ID**: TC-001.1
- **Steps**: `kf.x = np.zeros(6)`
- **Input**: None
- **Expected**: `[0,0,0,0,0,0]`
- **Actual**: ✅ `[0,0,0,0,0,0]`
- **Pass/Fail**: ✅ PASS
- **Coverage**: SF-EKF-001

#### TC-001.2: Covariance Matrix Initialization
- **Test ID**: TC-001.2
- **Steps**: `kf.P *= 10`
- **Input**: None
- **Expected**: Diagonal = 10.0
- **Actual**: ✅ Diagonal = 10.0
- **Pass/Fail**: ✅ PASS
- **Coverage**: SF-EKF-001

### TC-002: Motion Model Unit Tests

#### TC-002.1: Straight Motion
- **Test ID**: TC-002.1
- **Initial State**: `[0,0,10,0,0,0]`
- **dt**: 0.1
- **Expected Δx**: 1.0
- **Actual**: ✅ 1.000
- **Tolerance**: ±0.01
- **Pass/Fail**: ✅ PASS (SF-EKF-002)

#### TC-002.2: Turning Motion
- **Test ID**: TC-002.2
- **Initial State**: `[0,0,10,0,0,1]`
- **dt**: 0.1
- **Expected yaw**: 0.1 rad
- **Actual**: ✅ 0.1000
- **Tolerance**: ±0.001
- **Pass/Fail**: ✅ PASS (SF-EKF-002)

### TC-003: End-to-End Fusion (Your Main Test)

**Prerequisites**: 
- CARLA Simulator running
- Manual control active (vehicle driving)

#### TC-003.1: Straight Drive Scenario
- **Test ID**: TC-003.1
- **Scenario**: Straight drive
- **Duration**: 100 steps
- **GPS RMSE**: _______ m
- **EKF RMSE**: _______ m
- **Improvement %**: _______ %
- **Pass Criteria**: ≥30% better
- **Status**: 🔄 RUN TO VERIFY

#### TC-003.2: Turns + Curves Scenario
- **Test ID**: TC-003.2
- **Scenario**: Turns + curves
- **Duration**: 100 steps
- **GPS RMSE**: _______ m
- **EKF RMSE**: _______ m
- **Improvement %**: _______ %
- **Pass Criteria**: ≥25% better
- **Status**: 🔄 RUN TO VERIFY

#### TC-003.3: GPS Noise Only Scenario
- **Test ID**: TC-003.3
- **Scenario**: GPS noise only (σ=3.0m)
- **Duration**: 100 steps
- **GPS RMSE**: _______ m
- **EKF RMSE**: _______ m
- **Improvement %**: _______ %
- **Pass Criteria**: ≥30% better
- **Status**: 🔄 RUN TO VERIFY

### TC-004: Performance Metrics

#### TC-004.1: RMSE Improvement
- **Metric ID**: TC-004.1
- **Requirement**: RMSE Improvement ≥30%
- **Measured**: _______ %
- **Pass/Fail**: _______

#### TC-004.2: Max Position Error
- **Metric ID**: TC-004.2
- **Requirement**: Max Position Error <5m
- **Measured**: _______ m
- **Pass/Fail**: _______

#### TC-004.3: Yaw Drift
- **Metric ID**: TC-004.3
- **Requirement**: Yaw Drift <10° (1000 steps)
- **Measured**: _______ °
- **Pass/Fail**: _______

### TC-005: Stress Test

#### TC-005.1: 10Hz Processing
- **Test ID**: TC-005.1
- **Requirement**: Process @ 10Hz (100ms per iteration)
- **Avg Processing Time**: _______ ms
- **Max Processing Time**: _______ ms
- **Pass/Fail**: _______
- **Coverage**: SF-EKF-005

### TC-006: Position Error Validation

#### TC-006.1: Position Error Threshold
- **Test ID**: TC-006.1
- **Requirement**: Position error σ < 2.0m (1000 steps)
- **Mean Error**: _______ m
- **Std Error**: _______ m
- **Pass/Fail**: _______
- **Coverage**: SF-EKF-006

## Running the Tests

### 1. Unit Tests (No CARLA Required)
```bash
conda activate carla
cd /home/saad/dev/av-perception-portfolio
python -m pytest tests/week1/test_ekf_unit.py -v -s
```

**Expected Output**: 6/6 tests pass ✅

### 2. Integration Tests (CARLA Required)
```bash
# Terminal 1: Start CARLA
cd /home/saad/dev/carlaSim/CARLA_0.9.15
./CarlaUE4.sh -quality-level=Low

# Terminal 2: Start Driver
cd /home/saad/dev/carlaSim/CARLA_0.9.15/PythonAPI/examples
python manual_control.py

# Terminal 3: Run Tests
conda activate carla
cd /home/saad/dev/av-perception-portfolio
python -m pytest tests/week1/test_ekf_integration.py -v -s
```

### 3. Performance Tests (CARLA Required)
```bash
# With CARLA and manual_control running:
conda activate carla
python -m pytest tests/week1/test_ekf_performance.py -v -s
```

### 4. Run All Tests
```bash
conda activate carla
python tests/run_tests.py
```

## Test Results Summary

### Unit Tests (TC-001, TC-002)
- **Total**: 6 tests
- **Passed**: 6 ✅
- **Failed**: 0
- **Status**: ✅ ALL PASS

### Integration Tests (TC-003)
- **Total**: 3 tests
- **Status**: 🔄 Pending execution with CARLA

### Performance Tests (TC-004, TC-005, TC-006)
- **Total**: 5 tests
- **Status**: 🔄 Pending execution with CARLA

## Notes

1. **CARLA Dependency**: Tests TC-003 through TC-006 require CARLA simulator and an active vehicle (manual_control.py).

2. **Test Duration**: 
   - Unit tests: ~1 second
   - Integration tests: ~5 minutes (100 steps × 3 scenarios)
   - Performance tests: ~10 minutes (1000 steps × multiple metrics)

3. **Driving Instructions**:
   - TC-003.1: Drive straight for consistent results
   - TC-003.2: Make turns and curves
   - TC-003.3: Any driving pattern (tests filter robustness)

4. **Filling Results**: After running tests, fill in the blank fields (marked with `_______`) in this document with actual measured values.

## Troubleshooting

### Common Issues

1. **"No module named pytest"**
   ```bash
   conda activate carla
   pip install pytest
   ```

2. **"CARLA not available"**
   - Ensure CARLA is running on localhost:2000
   - Check firewall settings

3. **"No vehicle found"**
   - Run `manual_control.py` before tests
   - Verify vehicle spawned successfully

4. **Test timeout**
   - Increase CARLA timeout in test files
   - Check system performance

## Continuous Integration

To integrate these tests into CI/CD:

```yaml
# .github/workflows/test.yml
name: EKF Tests
on: [push, pull_request]
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/week1/test_ekf_unit.py -v
```

---

# Week 2 Radar-LiDAR Fusion Test Documentation

## Test Suite Overview

This section covers the unit and integration tests for the multi-modal sensor fusion system.

## Test Cases

### TC-201: Module Unit Tests

#### TC-201.1: Radar Conversion
- **Test ID**: TC-201.1
- **Target**: `radar_utils.py`
- **Verification**: Polar coordinates $(r, \theta, v_{rel})$ accurately convert to Cartesian $(x, y, v_x, v_y)$.
- **Status**: ✅ PASS

#### TC-201.2: LiDAR Filtering
- **Test ID**: TC-201.2
- **Target**: `lidar_utils.py`
- **Verification**: Height range filtering correctly removes ground/sky points and projects to BEV.
- **Status**: ✅ PASS

#### TC-201.3: Fusion Weighting
- **Test ID**: TC-201.3
- **Target**: `tracking.py`
- **Verification**: Weighted fusion correctly applies 70/30 LiDAR-to-radar ratio.
- **Status**: ✅ PASS

### TC-202: CARLA Integration Flow

#### TC-202.1: Sensor Attachment & Data Flow
- **Test ID**: TC-202.1
- **Scenario**: Spawn vehicle and attach sync sensors.
- **Verification**: Callbacks capture data and populate queues for both radar and LiDAR concurrently.
- **Status**: 🔄 RUN TO VERIFY

## Running the Tests

### 1. Week 1 Tests
```bash
python tests/week1/run_tests.py
```

### 2. Week 2 Tests
```bash
python tests/week2/run_tests.py
```
