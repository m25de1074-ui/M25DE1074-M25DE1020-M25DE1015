"""Tests for the speech pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from speech_pipeline.config import Config
from speech_pipeline.models import SpeakerSegment, TranscriptionResult
from speech_pipeline.audio_utils import AudioProcessor


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.whisper_model == "base"
        assert config.pyannote_model == "pyannote/speaker-diarization-3.1"
        assert config.output_format == "srt"
        assert config.speaker_labels == "Speaker"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Should raise error without HF token
        with pytest.raises(ValueError, match="Hugging Face token is required"):
            config.validate()
        
        # Should pass with token
        config.huggingface_token = "test_token"
        config.validate()  # Should not raise
        
        # Should raise error with invalid model
        config.whisper_model = "invalid_model"
        with pytest.raises(ValueError, match="Invalid Whisper model"):
            config.validate()


class TestSpeakerSegment:
    """Test SpeakerSegment model."""
    
    def test_segment_creation(self):
        """Test creating a speaker segment."""
        segment = SpeakerSegment(
            start=0.0,
            end=5.0,
            speaker="Speaker 1",
            text="Hello world",
            confidence=0.95
        )
        
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.duration == 5.0
        assert segment.speaker == "Speaker 1"
        assert segment.text == "Hello world"
        assert segment.confidence == 0.95
    
    def test_segment_to_dict(self):
        """Test converting segment to dictionary."""
        segment = SpeakerSegment(
            start=1.0,
            end=3.0,
            speaker="Speaker A",
            text="Test text",
            confidence=0.8
        )
        
        expected = {
            "start": 1.0,
            "end": 3.0,
            "speaker": "Speaker A",
            "text": "Test text",
            "confidence": 0.8,
            "duration": 2.0
        }
        
        assert segment.to_dict() == expected


class TestTranscriptionResult:
    """Test TranscriptionResult model."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample transcription result."""
        segments = [
            SpeakerSegment(0.0, 2.0, "Speaker 1", "Hello there", 0.9),
            SpeakerSegment(2.0, 4.0, "Speaker 2", "How are you?", 0.85),
            SpeakerSegment(4.0, 6.0, "Speaker 1", "I'm doing well", 0.92)
        ]
        
        return TranscriptionResult(
            segments=segments,
            total_duration=6.0,
            speakers=["Speaker 1", "Speaker 2"]
        )
    
    def test_to_srt(self, sample_result):
        """Test SRT format generation."""
        srt_content = sample_result.to_srt()
        
        assert "1" in srt_content
        assert "00:00:00,000 --> 00:00:02,000" in srt_content
        assert "Speaker 1: Hello there" in srt_content
        assert "Speaker 2: How are you?" in srt_content
    
    def test_to_vtt(self, sample_result):
        """Test WebVTT format generation."""
        vtt_content = sample_result.to_vtt()
        
        assert "WEBVTT" in vtt_content
        assert "00:00:00.000 --> 00:00:02.000" in vtt_content
        assert "Speaker 1: Hello there" in vtt_content
    
    def test_to_json(self, sample_result):
        """Test JSON format generation."""
        import json
        
        json_content = sample_result.to_json()
        data = json.loads(json_content)
        
        assert data["total_duration"] == 6.0
        assert len(data["speakers"]) == 2
        assert len(data["segments"]) == 3
        assert data["segments"][0]["text"] == "Hello there"
    
    def test_speaker_stats(self, sample_result):
        """Test speaker statistics calculation."""
        stats = sample_result.get_speaker_stats()
        
        assert "Speaker 1" in stats
        assert "Speaker 2" in stats
        
        # Speaker 1 has 4 seconds total (segments: 0-2, 4-6)
        assert stats["Speaker 1"]["total_time"] == 4.0
        assert abs(stats["Speaker 1"]["percentage"] - 66.67) < 0.1
        assert stats["Speaker 1"]["segments_count"] == 2
        
        # Speaker 2 has 2 seconds total (segment: 2-4)
        assert stats["Speaker 2"]["total_time"] == 2.0
        assert abs(stats["Speaker 2"]["percentage"] - 33.33) < 0.1
        assert stats["Speaker 2"]["segments_count"] == 1


class TestAudioProcessor:
    """Test audio processing utilities."""
    
    def test_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(sample_rate=22050)
        assert processor.sample_rate == 22050
    
    def test_preprocess_for_whisper(self):
        """Test audio preprocessing for Whisper."""
        processor = AudioProcessor()
        
        # Test with normal audio
        audio_data = np.array([0.5, -0.3, 0.8, -0.2], dtype=np.float32)
        processed = processor.preprocess_for_whisper(audio_data)
        
        assert processed.dtype == np.float32
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_extract_segment(self):
        """Test audio segment extraction."""
        processor = AudioProcessor()
        
        # Create 5 seconds of audio at 16kHz
        sample_rate = 16000
        audio_data = np.random.randn(5 * sample_rate).astype(np.float32)
        
        # Extract 1-3 second segment
        segment = processor.extract_segment(audio_data, 1.0, 3.0, sample_rate)
        
        expected_length = 2 * sample_rate  # 2 seconds
        assert len(segment) == expected_length
    
    def test_validate_audio_format(self):
        """Test audio format validation."""
        processor = AudioProcessor()
        
        # Test with non-existent file
        assert not processor.validate_audio_format("nonexistent.wav")
        
        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            assert not processor.validate_audio_format(tmp.name)


@pytest.mark.integration
class TestSpeechPipeline:
    """Integration tests for the speech pipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Config()
        config.huggingface_token = "test_token"
        config.whisper_model = "base"
        return config
    
    @patch('speech_pipeline.pipeline.PyannnotePipeline.from_pretrained')
    @patch('speech_pipeline.pipeline.WhisperModel')
    def test_pipeline_initialization(self, mock_whisper_cls, mock_pyannote_from_pretrained, mock_config):
        """Test pipeline initialization."""
        from speech_pipeline.pipeline import SpeechPipeline
        
        # Mock the models
        mock_whisper_instance = Mock()
        mock_whisper_instance.transcribe.return_value = ([], {})
        mock_whisper_cls.return_value = mock_whisper_instance
        mock_pyannote_from_pretrained.return_value = Mock()
        
        pipeline = SpeechPipeline(config=mock_config)
        
        assert pipeline.config == mock_config
        assert pipeline.whisper_model is not None
        assert pipeline.diarization_pipeline is not None
    
    def test_get_model_info(self, mock_config):
        """Test model information retrieval."""
        with patch('speech_pipeline.pipeline.WhisperModel') as mock_whisper_cls, \
             patch('speech_pipeline.pipeline.PyannnotePipeline.from_pretrained') as mock_pyannote_from_pretrained:
            mock_whisper_instance = Mock()
            mock_whisper_instance.transcribe.return_value = ([], {})
            mock_whisper_cls.return_value = mock_whisper_instance
            mock_pyannote_from_pretrained.return_value = Mock()
            
            from speech_pipeline.pipeline import SpeechPipeline
            pipeline = SpeechPipeline(config=mock_config)
            
            info = pipeline.get_model_info()
            
            assert "whisper_model" in info
            assert "pyannote_model" in info
            assert "device" in info
            assert "sample_rate" in info


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_pipeline.py -v
    pytest.main([__file__, "-v"])