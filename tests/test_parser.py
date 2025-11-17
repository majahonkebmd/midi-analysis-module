# tests/test_parser.py
import pytest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.midi_parser import MIDIParser

class TestMIDIParser:
    def test_parser_initialization(self):
        """Test that parser initializes correctly."""
        parser = MIDIParser()
        assert parser.parsed_data == {}
    
    def test_parser_has_parse_method(self):
        """Test that parser has the required parse_midi method."""
        parser = MIDIParser()
        assert hasattr(parser, 'parse_midi')
        assert callable(parser.parse_midi)