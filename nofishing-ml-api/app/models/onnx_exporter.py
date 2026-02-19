"""
ONNX Exporter - Export PyTorch model to ONNX format
"""
import logging
import os

import torch

from app.config import MODEL_PATH, ONNX_MODEL_PATH
from models.phishing_classifier import PhishingClassifier

logger = logging.getLogger(__name__)


def export_to_onnx(model_path: str = None, output_path: str = None):
    """
    Export trained model to ONNX format for optimized inference

    Args:
        model_path: Path to trained PyTorch model (.pt file)
        output_path: Path for output ONNX model (.onnx file)
    """
    model_path = model_path or MODEL_PATH
    output_path = output_path or ONNX_MODEL_PATH

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False

    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = PhishingClassifier()
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        # Create dummy input (batch_size=1, feature_dim=50)
        dummy_input = torch.randn(1, 50)

        # Export to ONNX
        logger.info(f"Exporting to ONNX: {output_path}")

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['lexical_features'],
            output_names=['phishing_probability'],
            dynamic_axes={
                'lexical_features': {0: 'batch_size'},
                'phishing_probability': {0: 'batch_size'}
            },
            verbose=False
        )

        logger.info(f"Successfully exported model to {output_path}")

        # Verify the exported model
        import onnx
        import onnxruntime as ort

        # Load and check ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")

        # Test inference
        ort_session = ort.InferenceSession(output_path)
        inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        outputs = ort_session.run(None, inputs)
        logger.info(f"Test inference output: {outputs[0][0][0]:.4f}")

        return True

    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    export_to_onnx()
