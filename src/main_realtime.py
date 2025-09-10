#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire Risk Assessment AI Agent - Main Application
**Directly evaluate multiple cameras and generate technical report**
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fire_risk_agent import FireRiskAssessmentAgent
from data_loader import DataLoader
from report_generator import ReportGenerator

# Logs
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'fire_risk_agent.log')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

class FireRiskAssessmentApp:
    """Wildfire risk assessment application"""
    
    def __init__(self):
        """Initialize"""
        self.agent = FireRiskAssessmentAgent()
        self.data_loader = DataLoader()
        self.report_generator = ReportGenerator()
        
        logger.info("Fire Risk Assessment Application initialized")
    
    def run(self, camera_filename: str = 'camera_monitoring_dataset.jsonl'):
        """
        Run wildfire risk assessment
        
        Args:
            camera_filename: Camera data filename
        """
        logger.info("Starting fire risk assessment...")
        
        try:
            # 1. Load camera data
            logger.info("Loading camera data...")
            assessment_data = self.data_loader.get_assessment_data(camera_filename)
            
            if not assessment_data:
                logger.error("Failed to load camera data")
                return False
            
            logger.info(f"Loaded {len(assessment_data)} camera clusters for assessment")
            
            # 2. AI assessment
            logger.info("Performing AI risk assessment and generating reports...")
            successful_reports = 0
            
            for i, single_assessment_data in enumerate(assessment_data, 1):
                try:
                    logger.info(f"Processing progress: {i}/{len(assessment_data)}")
                    
                    # AI assess single data point
                    result = self.agent.assess_fire_risk(single_assessment_data)
                    
                    # Generate report immediately
                    json_file = self.report_generator.generate_complete_report(
                        result, 
                        self.agent.model_name
                    )
                    
                    if json_file:
                        successful_reports += 1
                        logger.info(f"Report generated: {json_file}")
                    
                    # Avoid excessive API calls
                    if i < len(assessment_data):
                        import time
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error processing assessment {i}: {str(e)}")
                    continue
            
            logger.info(f"Assessment and report generation completed: {successful_reports}/{len(assessment_data)} reports generated")
            logger.info(f"Reports saved to: {os.path.abspath(self.report_generator.output_dir)}")
            
            print("\n✅ Fire risk assessment completed successfully!")
            print(f"📊 Assessed {len(assessment_data)} camera clusters")
            print(f"📄 Generated {successful_reports} technical reports")
            print(f"📁 Reports location: {os.path.abspath(self.report_generator.output_dir)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Assessment failed: {str(e)}")
            print(f"\n❌ Assessment failed: {str(e)}")
            return False

def main():
    """main fuction"""
    try:
        logger.info("=== Fire Risk Assessment Application Started ===")
        
        # Initialize and run application
        app = FireRiskAssessmentApp()
        success = app.run()
        
        if success:
            logger.info("Application completed successfully")
            sys.exit(0)
        else:
            logger.error("Application completed with errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n⏹️ Assessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        print(f"\n❌ Application failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()