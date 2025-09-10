#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire Risk Assessment AI Agent - Main AI Agent Class (Enhanced with HF Model)
Support Hugging Face gpt-oss-20b model and camera clustering analysis
"""

import json
import time
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FireRiskAssessmentAgent:
    """Fire Risk Assessment AI Agent (Enhanced with HF Model)"""

    # --------------------- Hugging Face model -----------------------
    def __init__(self, model_name: str = "openai/gpt-oss-20b", device: str = "auto"):
        """
        Initialize AI agent
        
        Args:
            model_name: Hugging Face model name
            device: device select ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        
        # Generation parameter configuration
        self.generation_params = {
            "max_new_tokens": 2200,     
            "temperature": 0.1,        
            "top_k": 40,              
            "top_p": 0.9,             
            "do_sample": True,         
            "pad_token_id": None,     
            "eos_token_id": None,      
            "repetition_penalty": 1.1, 
        }
        
        # --------------------- Prompt -----------------------
        self.prompt_template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'src', 'prompts', 'fire_assessment_prompt_1.txt'
        )
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Initialize model and tokenizer
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize Hugging Face model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info("This may take a few minutes for the first time...")
            
            # Set model loading parameters
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": self.device if self.device != "auto" else "auto",
                "trust_remote_code": True,
            }
            
            # Check if GPU is available
            if torch.cuda.is_available() and self.device in ["auto", "cuda"]:
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
                model_kwargs["device_map"] = "auto"
            else:
                logger.info("Using CPU for inference")
                model_kwargs["device_map"] = None
                model_kwargs["torch_dtype"] = torch.float32  # Use float32 for CPU inference
            
            # Load using pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                **model_kwargs
            )
            
            # Set pad_token_id and eos_token_id
            if hasattr(self.pipe.tokenizer, 'pad_token_id') and self.pipe.tokenizer.pad_token_id is not None:
                self.generation_params["pad_token_id"] = self.pipe.tokenizer.pad_token_id
            else:
                # If no pad_token，use eos_token
                self.generation_params["pad_token_id"] = self.pipe.tokenizer.eos_token_id
            
            self.generation_params["eos_token_id"] = self.pipe.tokenizer.eos_token_id
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.error("Please ensure you have sufficient GPU memory or try using CPU")
            raise
    
    def _load_prompt_template(self) -> str:
        """
        Load prompt template from file
        
        Returns:
            Prompt template string
        """
        if not os.path.exists(self.prompt_template_path):
            raise FileNotFoundError(f"Prompt template file does not exist: {self.prompt_template_path}")
        
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        logger.info("Prompt template loaded successfully!")
        return template
    
    def update_generation_params(self, **kwargs) -> None:
        """
        Update model generation parameters
        
        Args:
            **kwargs: Generation parameters dictionary
        """
        for key, value in kwargs.items():
            if key in self.generation_params:
                self.generation_params[key] = value
                logger.info(f"Parameter {key} has been updated to {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
    
    def _format_prompt_parameters(self, assessment_data: Dict) -> Dict[str, str]:
        """
        Format prompt parameters
        
        Args:
            assessment_data: Unified evaluation data from data_loader
            
        Returns:
            Formatted parameter dictionary
        """
        location = assessment_data.get('location', {})
        coordinates = location.get('coordinates', {})
        camera_assessment = assessment_data.get('camera_assessment', {})
        weather = assessment_data.get('weather', {})
        fire_risk_data = assessment_data.get('fire_risk_data', {})
        fire_stations = assessment_data.get('fire_stations', {})
        
        # Format location information
        location_info = f"{location.get('county', 'Unknown')} - {location.get('display_name', 'Unknown')}"
        
        # Format coordinates
        coordinates_str = f"{coordinates.get('lat', 0):.6f}, {coordinates.get('lon', 0):.6f}"
        
        # Format administrative information
        admin_info = f"City: {location.get('city', 'Unknown')}, County: {location.get('county', 'Unknown')}, State: California"
        
        # Format land type
        land_type = location.get('land_type', 'Unknown')
        
        # Format FHSZ risk
        fhsz_risk_level = f"{fire_risk_data.get('fhsz_risk_level', 'Unknown')} (Level {fire_risk_data.get('fhsz_risk', 'Unknown')})"
        
        # Format fire history
        fire_history = f"{fire_risk_data.get('fire_count', 0)} historical incidents, Dates: {', '.join(fire_risk_data.get('fire_dates', []) or ['None'])}"
        
        # Format weather data
        temp = weather.get('temperature', {})
        wind = weather.get('wind_speed', {})
        weather_data = f"Temperature: {temp.get('fahrenheit', 'N/A')}°F ({temp.get('celsius', 'N/A')}°C), Humidity: {weather.get('humidity', 'N/A')}%, Wind: {wind.get('mph', 'N/A')}mph ({wind.get('mps', 'N/A')}m/s), Conditions: {weather.get('weather_description', 'Unknown')}"
        
        # Format camera analysis
        camera_analysis = f"{camera_assessment.get('total_cameras', 0)} cameras deployed, {camera_assessment.get('cameras_detecting_fire', 0)} detecting fire, {camera_assessment.get('cameras_detecting_smoke', 0)} detecting smoke, Summary: {camera_assessment.get('detection_summary', 'No data')}"
        
        # Format fire resources
        stations_detail = fire_stations.get('stations_detail', [])
        if stations_detail:
            nearest_stations = []
            for station in stations_detail[:3]:  # Take the 3 nearest
                name = station.get('name', 'Unknown')
                distance = station.get('distance_km', 'N/A')
                phone = station.get('phone', 'No phone')
                nearest_stations.append(f"{name} ({distance}km, {phone})")
            resources_info = f"{fire_stations.get('station_count', 0)} fire stations available. Nearest: {'; '.join(nearest_stations)}"
        else:
            resources_info = "No fire station data available"
        
        return {
            'location_info': location_info,
            'coordinates': coordinates_str,
            'admin_info': admin_info,
            'land_type': land_type,
            'fhsz_risk_level': fhsz_risk_level,
            'fire_history': fire_history,
            'weather_data': weather_data,
            'camera_analysis': camera_analysis,
            'resources_info': resources_info
        }
    
    def _create_prompt(self, assessment_data: Dict) -> str:
        """
        Create prompt for wildfire risk assessment
        
        Args:
            assessment_data: Unified evaluation data dictionary
            
        Returns:
            Constructed prompt string
        """
        # Format parameters
        params = self._format_prompt_parameters(assessment_data)
        
         # Check all placeholders in the template
        import re
        placeholders = re.findall(r'\{(\w+)\}', self.prompt_template)
        
        # 找出缺失的参数
        missing = [p for p in placeholders if p not in params]
        if missing:
            print(f"Missing parameters: {missing}")
        
        logger.info(f"Prompt parameters prepared with {len(params)} parameters")
        
        try:
            formatted_prompt = self.prompt_template.format(**params)
            logger.debug(f"Prompt generated successfully, length: {len(formatted_prompt)} characters")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Prompt parameter substitution failed, missing parameter: {e}")
            logger.error(f"Template content preview: {self.prompt_template[:200]}...")
            logger.error(f"Available parameters: {list(params.keys())}")
            raise
        except Exception as e:
            logger.error(f"Prompt creation failed: {str(e)}")
            raise
    
    def _call_hf_model(self, prompt: str) -> Optional[str]:
        """
        Call Hugging Face model to get response
        
        Args:
            prompt: input prompt
            
        Returns:
            Model generated response text
        """
        try:
            logger.info("Calling Hugging Face model for analysis...")
            start_time = time.time()
    
            # Create system prompt, set low inference level for concise output
            system_prompt = """You are a fire risk assessment AI. Reasoning: low
    Provide only JSON output as requested. Be concise and direct."""
            
            # Prepare message format, let Transformers automatically apply harmony format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Call model
            outputs = self.pipe(
                messages,
                max_new_tokens=self.generation_params["max_new_tokens"],
                temperature=self.generation_params["temperature"],
                top_k=self.generation_params["top_k"],
                top_p=self.generation_params["top_p"],
                do_sample=self.generation_params["do_sample"],
                pad_token_id=self.generation_params["pad_token_id"],
                eos_token_id=self.generation_params["eos_token_id"],
                repetition_penalty=self.generation_params["repetition_penalty"]
            )
            
            end_time = time.time()
            logger.info(f"Model response time: {end_time - start_time:.2f} seconds")
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    for msg in reversed(generated_text):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            response = msg.get("content", "")
                            break
                    else:
                        response = generated_text[-1].get("content", "") if generated_text else ""
                else:
                    response = str(generated_text)
                    
                    if "assistant" in response:
                        lines = response.split('\n')
                        assistant_content = []
                        in_assistant = False
                        
                        for line in lines:
                            if '"role": "assistant"' in line or 'role: assistant' in line:
                                in_assistant = True
                                continue
                            elif '"role":' in line and in_assistant:
                                break
                            elif in_assistant and '"content":' in line:
                                # Extract value of content field
                                content_start = line.find('"content":')
                                if content_start != -1:
                                    content_part = line[content_start + 10:].strip()
                                    if content_part.startswith('"'):
                                        content_part = content_part[1:]
                                    if content_part.endswith('",') or content_part.endswith('"'):
                                        content_part = content_part.rstrip('",')
                                    assistant_content.append(content_part)
                            elif in_assistant and line.strip():
                                assistant_content.append(line.strip())
                        
                        if assistant_content:
                            response = ' '.join(assistant_content)
                
                # Logs
                logger.info(f"Extracted response preview: {response[:200] if response else 'None'}...")
                return response.strip() if response else None
            else:
                logger.error("No output received from model")
                return None
            
        except Exception as e:
            logger.error(f"Model call exception: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return None
    
    def _parse_assessment_response(self, response: str, assessment_data: Dict) -> Dict:
        """Parse evaluation response from AI model"""
        
        logger.info(f"Raw AI response (first 300 chars): {response[:300] if response else 'None'}")
        
        # Default result
        result = {
            'risk_score': 3,
            'risk_level': 'Moderate', 
            'reasoning': 'AI assessment completed',
            'confidence': 'Medium',
            'emergency_recommendations': [
                "Monitor fire conditions closely",
                "Prepare emergency response teams",
                "Maintain communication with fire stations"
            ],
            'raw_response': response or ''
        }
        
        if not response or not response.strip():
            logger.warning("Empty response from AI model")
            return result
        
        try:
            # Use enhanced JSON extraction
            json_str = self._extract_json_safely(response.strip())
            if not json_str:
                logger.warning("No valid JSON found in response")
                return result
    
            logger.info(f"Extracted JSON length: {len(json_str)} characters")
            
            # Basic cleaning
            json_str = json_str.replace('\\n', ' ').replace('\\"', '"')
            
            # Attempt parsing
            parsed = json.loads(json_str)
            logger.info("JSON parsing successful!")
            
            # Extract data
            ai_assessment = parsed.get('ai_assessment', {})
            
            if ai_assessment:
                result['risk_score'] = ai_assessment.get('risk_score', result['risk_score'])
                result['risk_level'] = ai_assessment.get('risk_level', result['risk_level'])
                result['reasoning'] = ai_assessment.get('reasoning', result['reasoning'])
                result['confidence'] = ai_assessment.get('confidence', result['confidence'])

                # Handle detection_confidence field name variants
                detection_conf = (ai_assessment.get('detection_confidence') or 
                                 ai_assessment.get('detector_confidence') or 
                                 'Normal')
                result['detection_confidence'] = detection_conf
                
                logger.info(f"Successfully extracted: Level={result['risk_level']}, Score={result['risk_score']}")
                logger.info(f"Detection confidence: {detection_conf}")
            
            # Extract suggestions
            emergency_recs = parsed.get('emergency_recommendations', [])
            if emergency_recs and isinstance(emergency_recs, list) and len(emergency_recs) > 0:
                clean_recs = [rec.strip() for rec in emergency_recs if isinstance(rec, str) and len(rec.strip()) > 10]
                if clean_recs:
                    result['emergency_recommendations'] = self._enhance_emergency_recs(clean_recs, assessment_data)
            
            monitoring_recs = parsed.get('monitoring_recommendations', [])
            if monitoring_recs and isinstance(monitoring_recs, list):
                clean_monitoring = [rec.strip() for rec in monitoring_recs if isinstance(rec, str) and len(rec.strip()) > 10]
                result['monitoring_recommendations'] = clean_monitoring if clean_monitoring else []
            else:
                result['monitoring_recommendations'] = []
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed even after repair: {e}")
            logger.warning(f"Final JSON attempt: {json_str[:300] if 'json_str' in locals() else 'None'}")
        except Exception as e:
            logger.warning(f"Unexpected error in parsing: {e}")
    
        return result
    

    def _extract_json_safely(self, response: str) -> str:
        """JSON extraction"""
        cleaned = response.strip()
        json_start = cleaned.find('{')
        
        logger.info(f"Debug: response length = {len(cleaned)}")
        logger.info(f"Debug: json_start = {json_start}")
        
        if json_start == -1:
            return None
        
        # Try to find complete JSON
        brace_count = 0
        in_string = False
        escaped = False
        
        for i in range(json_start, len(cleaned)):
            char = cleaned[i]
            
            if escaped:
                escaped = False
                continue
            
            if char == '\\':
                escaped = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        result = cleaned[json_start:i+1]
                        logger.info(f"Debug: found complete JSON")
                        return result
        
        # If complete JSON is not found, try intelligent repair
        logger.info("Debug: JSON appears incomplete, attempting repair")
        return self._repair_incomplete_json(cleaned[json_start:])
        
    
    def assess_fire_risk(self, assessment_data: Dict) -> Dict:
        """
        Execute wildfire risk assessment
        
        Args:
            assessment_data: Unified evaluation data from data_loader
            
        Returns:
            Dictionary containing risk assessment results
        """
        location_name = assessment_data.get('location', {}).get('display_name', 'Unknown')
        logger.info(f"Starting risk assessment for location: {location_name}")
        
        # Create prompt
        prompt = self._create_prompt(assessment_data)
        
        # Call Hugging Face model
        response = self._call_hf_model(prompt)
        
        if response is None:
            return {
                'ai_assessment': {
                    'risk_score': 1,
                    'risk_level': 'Assessment Failed',
                    'reasoning': 'Unable to get response from AI model',
                    'confidence': 'Low'
                },
                'emergency_recommendations': [],
                'monitoring_recommendations': [],
                'timestamp': datetime.now().isoformat(),
                'sensor_data': assessment_data,
                'model_params': self.generation_params.copy(),
                'raw_ai_response': ''
            }
        
        # Parse response
        # assessment_result = self._parse_assessment_response(response)
        assessment_result = self._parse_assessment_response(response, assessment_data)

        # Construct complete result
        complete_result = {
            'ai_assessment': {
                'risk_score': assessment_result['risk_score'],
                'risk_level': assessment_result['risk_level'],
                'reasoning': assessment_result['reasoning'],
                'confidence': assessment_result['confidence']
            },
            'emergency_recommendations': assessment_result['emergency_recommendations'],
            'monitoring_recommendations': assessment_result.get('monitoring_recommendations', []),
            'timestamp': datetime.now().isoformat(),
            'sensor_data': assessment_data,
            'model_params': self.generation_params.copy(),
            'raw_ai_response': assessment_result.get('raw_response', '')
        }
        
        logger.info(f"Assessment completed - Risk level: {assessment_result['risk_level']} (Score: {assessment_result['risk_score']})")
        
        return complete_result


    def _repair_incomplete_json(self, json_fragment: str) -> str:
        """Fix incomplete JSON and handle missing field values"""
        try:
            logger.info("Attempting to repair incomplete JSON...")
            
            if json_fragment.strip().endswith(':'):
                if '"confidence":' in json_fragment and json_fragment.strip().endswith('"confidence":'):
                    json_fragment += '"Medium"'
                elif '"detection_confidence":' in json_fragment and json_fragment.strip().endswith('"detection_confidence":'):
                    json_fragment += '"Normal"'
                elif '"detector_confidence":' in json_fragment and json_fragment.strip().endswith('"detector_confidence":'):
                    json_fragment += '"Normal"'
            
            open_braces = json_fragment.count('{')
            close_braces = json_fragment.count('}')
            missing_braces = open_braces - close_braces
            
            if '"ai_assessment":' in json_fragment and missing_braces > 0:
                repaired = json_fragment.rstrip(',').rstrip()
                
                # Ensure confidence field exists
                if '"confidence":' in repaired and not ('"confidence":"' in repaired or '"confidence": "' in repaired):
                    repaired += ',"confidence":"Medium"'
                elif '"confidence"' not in repaired:
                    repaired += ',"confidence":"Medium"'
                
                # Ensure detection_confidence field exists
                if '"detection_confidence"' not in repaired and '"detector_confidence"' not in repaired:
                    repaired += ',"detection_confidence":"Normal"'
                
                # Close ai_assessment object
                if missing_braces >= 1:
                    repaired += '}'
                    missing_braces -= 1
            else:
                repaired = json_fragment
            
            if '"emergency_recommendations":' not in repaired:
                if not repaired.rstrip().endswith(','):
                    repaired += ','
                repaired += '"emergency_recommendations":["Monitor conditions closely","Prepare emergency response"]'
            
            if '"monitoring_recommendations":' not in repaired:
                if not repaired.rstrip().endswith(','):
                    repaired += ','
                repaired += '"monitoring_recommendations":["Regular monitoring recommended","Weather tracking"]'
            
            # Ensure outermost JSON ends correctly
            open_braces = repaired.count('{')
            close_braces = repaired.count('}')
            remaining_braces = open_braces - close_braces
            
            if remaining_braces > 0:
                repaired += '}' * remaining_braces
            
            # Basic format cleaning
            repaired = repaired.replace(',,', ',').replace(',}', '}')
            
            logger.info(f"JSON repair completed, length: {len(repaired)}")
            logger.info(f"Repaired JSON preview: {repaired[:200]}...")
            
            # Validate repaired JSON
            try:
                json.loads(repaired)
                logger.info("Repaired JSON is valid!")
                return repaired
            except json.JSONDecodeError as e:
                logger.warning(f"Repaired JSON still invalid: {e}")
                return self._create_minimal_fallback_json()
            
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return self._create_minimal_fallback_json()
    
    def _create_minimal_fallback_json(self) -> str:
        """Create a minimal usable JSON as fallback"""
        return '''
    {
      "ai_assessment": {
        "risk_level": "Moderate",
        "risk_score": 3,
        "reasoning": "Assessment completed with partial data",
        "confidence": "Medium",
        "detection_confidence": "Normal"
      },
      "emergency_recommendations": [
        "Monitor conditions closely",
        "Prepare emergency response"
      ],
      "monitoring_recommendations": [
        "Regular monitoring recommended",
        "Weather tracking"
      ]
    }
    '''.strip()
        
    
    def batch_assess(self, assessment_data_list: List[Dict]) -> List[Dict]:
        """
        Batch evaluate multiple data points
        
        Args:
            assessment_data_list: Evaluation data list
            
        Returns:
            Evaluation result list
        """
        results = []
        total = len(assessment_data_list)
        
        logger.info(f"Starting batch assessment of {total} data points")
        
        for i, assessment_data in enumerate(assessment_data_list, 1):
            logger.info(f"Processing: {i}/{total}")
            result = self.assess_fire_risk(assessment_data)
            results.append(result)
            
            # Avoid excessive memory usage, take appropriate breaks
            if i < total:
                time.sleep(0.5)
        
        logger.info("Batch assessment completed")
        return results
    
    def reload_prompt_template(self) -> bool:
        """
        Reload prompt template
        
        Returns:
            Whether reloaded successfully
        """
        try:
            new_template = self._load_prompt_template()
            self.prompt_template = new_template
            logger.info("Prompt template reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading prompt template: {str(e)}")
            return False

    def _enhance_emergency_recs(self, original_recs, assessment_data):
        """Use RAG to enhance emergency suggestions"""
        logger.info("=== RAG enhancement started ===")
        try:
            if not hasattr(self, 'emergency_rag'):
                logger.info("Initialize EmergencyRAG...")
                from emergency_rag import EmergencyRAG
                self.emergency_rag = EmergencyRAG()
            
            logger.info(f"RAG status: {self.emergency_rag.get_status()}")
            
            enhanced_recs = self.emergency_rag.enhance_recommendations(
                original_recs, 
                assessment_data
            )
            logger.info(f"RAG enhancement completed: {len(original_recs)} -> {len(enhanced_recs)}")
            return enhanced_recs
        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}")
            return original_recs
        
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        try:
            return {
                'model_name': self.model_name,
                'device': str(self.pipe.device) if hasattr(self.pipe, 'device') else 'unknown',
                'torch_dtype': str(self.pipe.model.dtype) if hasattr(self.pipe.model, 'dtype') else 'unknown',
                'generation_params': self.generation_params.copy(),
                'model_size': f"{self.pipe.model.num_parameters() / 1e9:.1f}B parameters" if hasattr(self.pipe.model, 'num_parameters') else 'unknown'
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}