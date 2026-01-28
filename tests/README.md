# Portfolio Test Suite

This directory contains the automated test suites for the perception projects, organized by week.

## Directory Structure

```text
tests/
├── week1/                  # GPS + IMU Fusion (EKF)
│   ├── test_ekf_unit.py
│   ├── test_ekf_integration.py
│   ├── test_ekf_performance.py
│   └── run_tests.py        # Week 1 Runner
├── week2/                  # Radar + LiDAR Fusion
│   ├── test_fusion_unit.py
│   ├── test_fusion_integration.py
│   └── run_tests.py        # Week 2 Runner
├── TEST_DOCUMENTATION.md   # Detailed requirements & TC mapping
└── README.md               # This file
```

## How to Run Tests

### Week 1: Extended Kalman Filter
```bash
conda activate carla
python tests/week1/run_tests.py
```

### Week 2: Radar-LiDAR Fusion
```bash
conda activate carla
python tests/week2/run_tests.py
```

## Test Types

1. **Unit Tests**: Test individual functions (conversions, filtering) in isolation. No CARLA required.
2. **Integration Tests**: Verify sensor attachments and data flow. **Requires CARLA simulator**.
3. **Performance Tests**: Quantitative validation of tracking accuracy. **Requires CARLA simulator**.

See `TEST_DOCUMENTATION.md` for full test case mapping and requirements coverage.
