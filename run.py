
"""
Main script to run the financial ML pipeline.
"""

import argparse
import logging
from pathlib import Path
import yaml

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.features.feature_pipeline import FeaturePipeline
from src.models.model_factory import ModelFactory
from src.strategy.trading_strategy import TradingStrategy
from src.strategy.risk_manager import RiskManager
from src.strategy.execution_manager import ExecutionManager
from src.strategy.portfolio_monitor import PortfolioMonitor
from src.reporting.report_generator import ReportGenerator
from src.utils.config import load_config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: str):
    """Run the financial ML pipeline."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        # Initialize components
        data_loader = DataLoader(config['data'])
        data_processor = DataProcessor(config['data'])
        feature_pipeline = FeaturePipeline(config['features'])
        model_factory = ModelFactory()

        # Load and process data
        logger.info("Loading data...")
        data = data_loader.fetch_data(
            config['pipeline']['target_symbol'],
            config['pipeline']['start_date'],
            config['pipeline']['end_date']
        )

        # Process data
        logger.info("Processing data...")
        processed_data = data_processor.process_data(data)

        # Generate features
        logger.info("Generating features...")
        features = feature_pipeline.fit_transform(processed_data)

        # Create and train model
        logger.info("Training model...")
        model = model_factory.create_model(
            config['model']['type'],
            config['model']['params']
        )
        model.fit(features['train'], features['target'])

        # Initialize trading components
        trading_strategy = TradingStrategy(config['strategy'], model)
        risk_manager = RiskManager(config['risk'])
        execution_manager = ExecutionManager(config['strategy'])
        portfolio_monitor = PortfolioMonitor(
            config['strategy'], execution_manager)

        # Generate trading signals
        logger.info("Generating trading signals...")
        signals = trading_strategy.generate_signals(
            features['test'],
            processed_data
        )

        # Apply risk management
        logger.info("Applying risk management...")
        adjusted_signals = risk_manager.adjust_signals(signals, processed_data)

        # Execute trades
        logger.info("Executing trades...")
        execution_manager.execute_signals(adjusted_signals)

        # Monitor portfolio
        logger.info("Monitoring portfolio...")
        portfolio_monitor.update()

        # Generate report
        logger.info("Generating report...")
        report_generator = ReportGenerator(config['reporting'])
        report_generator.generate_report()

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Financial ML Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)
