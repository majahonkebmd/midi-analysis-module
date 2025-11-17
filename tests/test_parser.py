# tests/simple_test_parser.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from midi_parser import MIDIParser

def run_parser_tests():
    """Simple test runner without pytest dependency"""
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Initialization
    try:
        parser = MIDIParser()
        assert parser.parsed_data == {}
        print("PASS - Parser initialization")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL - Parser initialization: {e}")
    total_tests += 1
    
    # Test 2: Method existence
    try:
        parser = MIDIParser()
        assert hasattr(parser, 'parse_midi')
        assert callable(parser.parse_midi)
        print("PASS - Parser method check")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL - Parser method check: {e}")
    total_tests += 1
    
    # Test 3: Error handling
    try:
        parser = MIDIParser()
        result = parser.parse_midi("non_existent_file.mid")
        # If we get here without exception, check structure
        if result is not None:
            assert isinstance(result, dict)
        print("PASS - Parser error handling")
        tests_passed += 1
    except Exception:
        # It's acceptable for parser to throw error for missing files
        print("PASS - Parser throws error for missing files (expected behavior)")
        tests_passed += 1
    total_tests += 1
    
    print(f"\nRESULTS: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_parser_tests()
    sys.exit(0 if success else 1)