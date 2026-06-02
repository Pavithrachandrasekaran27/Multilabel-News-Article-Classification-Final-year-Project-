"""
Run All Tests Script
Executes Unit, Integration, Validation, and System Tests
"""

import unittest
import time

def run_all_tests():
    print("\n" + "="*70)
    print("MULTI-LABEL NEWS CLASSIFICATION - TEST EXECUTION")
    print("="*70)

    start_time = time.time()

    # Discover all tests inside the 'tests' folder
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    end_time = time.time()
    duration = end_time - start_time

    # --------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------
    print("\n" + "="*70)
    print("TEST SUMMARY REPORT")
    print("="*70)

    print(f"Total Tests Run     : {result.testsRun}")
    print(f"Successful Tests    : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures            : {len(result.failures)}")
    print(f"Errors              : {len(result.errors)}")
    print(f"Execution Time (s)  : {duration:.2f}")

    if result.wasSuccessful():
        print("\nSTATUS: ALL TESTS PASSED SUCCESSFULLY")
    else:
        print("\nSTATUS: SOME TESTS FAILED — CHECK LOG ABOVE")

    print("="*70 + "\n")

if __name__ == "__main__":
    run_all_tests()
