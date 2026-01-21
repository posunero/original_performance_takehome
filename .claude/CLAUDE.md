# Project Instructions

## Performance Optimization Workflow

This project involves optimizing `perf_takehome.py` to minimize cycle count on a custom VLIW SIMD architecture.

### Changelog Requirement

**After every attempted optimization**, update `docs/CHANGELOG.md`:

1. **If successful**: Add a new version entry at the top of "Version History" with:
   - Version number and descriptive name
   - New cycle count and speedup
   - List of changes made
   - Key insights learned

2. **If failed**: Add an entry to the "Failed Experiments" section with:
   - What was attempted
   - The result (cycle count regression or correctness failure)
   - Why it failed (root cause analysis)

This documentation helps avoid repeating failed approaches and builds institutional knowledge about the architecture's performance characteristics.

### Testing

Always verify changes with:
```bash
python -m pytest tests/submission_tests.py -v
```

Key tests:
- `test_kernel_correctness` - Must pass (output matches reference)
- `test_kernel_speedup` - Must pass (speedup > 1x)
- Other tests are aspirational targets for deeper optimization
