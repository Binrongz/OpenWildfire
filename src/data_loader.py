#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader (Enhanced with Camera and Geographic Integration)
Read and process camera, geographic, and weather data, supporting multi-camera aggregation
"""

import pandas as pd
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from math import radians, cos, sin, sqrt, atan2
import sys
import re

# Add api_utils
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'api_utils'))

try:
    from weather_fetcher import get_weather
except ImportError:
    logging.warning("Cannot import weather_fetcher, weather functionality will be unavailable")
    def get_weather(lat, lon):
        return {"error": "Weather service unavailable"}

logger = logging.getLogger(__name__)

class DataLoader:
    """Data Loader Class (Enhanced)"""
    
    def __init__(self, data_dir: str = "data"):
        """
       Initialize data loader
        
        Args:
            data_dir: Data folder path
        """
        self.data_dir = data_dir
        self.static_geo_data = None
        self.camera_cluster_radius = 500  # meter
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two lat lon points (meters)
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance（meter）
        """
        R = 6371000  # Earth radius (meters)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    def load_camera_jsonl_data(self, filename: str = "camera_monitoring_dataset.jsonl") -> Optional[List[Dict]]:
        """
        Load camera data from JSONL file
        
        Args:
            filename: JSONL filename
            
        Returns:
            Camera data list, return None if loading fails
        """
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            if not os.path.exists(file_path):
                logger.error(f"Camera data file does not exist: {file_path}")
                return None
            
            camera_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        camera_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Successfully loaded {len(camera_data)} camera records from file: {filename}")
            return camera_data
            
        except Exception as e:
            logger.error(f"Failed to load camera data: {str(e)}")
            return None
    
    def cluster_cameras_by_distance(self, cameras: List[Dict], radius: float = 500) -> List[List[Dict]]:
        """
        Use optimized graph clustering algorithm to aggregate cameras by distance
        
        Args:
            cameras: Camera data list
            radius: Aggregation radius (meters)
            
        Returns:
            Camera clusters list
        """
        logger.info(f"Start aggregating camera data, aggregation radius: {radius}meters")
        
        clusters = []
        unprocessed = cameras.copy()
        
        while unprocessed:
            # Start a new cluster from an unprocessed camera
            seed = unprocessed.pop(0)
            cluster = [seed]
            
            # Iteratively add nearby cameras (graph connectivity clustering)
            changed = True
            while changed:
                changed = False
                for cam in unprocessed[:]:  # Iterate over a copy to avoid modification issues
                    # Check if the camera is within radius of any camera in the cluster
                    for cluster_cam in cluster:
                        distance = self.haversine_distance(
                            cam['lat'], cam['lon'],
                            cluster_cam['lat'], cluster_cam['lon']
                        )
                        if distance <= radius:
                            cluster.append(cam)
                            unprocessed.remove(cam)
                            changed = True
                            break
                    if changed:
                        break
            
            clusters.append(cluster)
        
        logger.info(f"Camera aggregation completed: {len(cameras)} cameras → {len(clusters)} clusters")
        
        # Summarize cluster information
        single_cam_clusters = sum(1 for cluster in clusters if len(cluster) == 1)
        multi_cam_clusters = len(clusters) - single_cam_clusters
        max_cluster_size = max(len(cluster) for cluster in clusters) if clusters else 0
        
        logger.info(f"Cluster stats: {single_cam_clusters} single-camera, {multi_cam_clusters} multi-camera clusters, max cluster size: {max_cluster_size}")
        
        return clusters
    
    def analyze_cluster_detection(self, cluster: List[Dict]) -> Dict:
        """
        Analyze detection status of camera clusters
        
        Args:
            cluster: Camera clusters
            
        Returns:
            Cluster detection analysis results
        """
        total_cameras = len(cluster)
        fire_cameras = [cam for cam in cluster if cam.get('detection_status') == 'fire']
        smoke_cameras = [cam for cam in cluster if cam.get('detection_status') == 'smoke']
        normal_cameras = [cam for cam in cluster if cam.get('detection_status') == 'normal']
        
        fire_count = len(fire_cameras)
        smoke_count = len(smoke_cameras)
        normal_count = len(normal_cameras)

        # confidence = "Medium"
        
        # Determine primary detection status
        if fire_count > 0:
            primary_status = "fire"
        elif smoke_count > 0:
            primary_status = "smoke"
        else:
            primary_status = "normal"
        
        # Calculate cluster center location
        center_lat = sum(cam['lat'] for cam in cluster) / total_cameras
        center_lon = sum(cam['lon'] for cam in cluster) / total_cameras
        
        # Generate detection summary
        detection_summary = self._generate_detection_summary(fire_count, smoke_count, normal_count, total_cameras)
        
        return {
            'cluster_center': {'lat': center_lat, 'lon': center_lon},
            'total_cameras': total_cameras,
            'cameras_detecting_fire': fire_count,
            'cameras_detecting_smoke': smoke_count,
            'cameras_normal': normal_count,
            'camera_ids': [cam['camera_id'] for cam in cluster],
            # 'detection_confidence': confidence,
            'primary_detection_status': primary_status,
            'detection_summary': detection_summary,
            'fire_detection_rate': fire_count / total_cameras,
            'smoke_detection_rate': smoke_count / total_cameras
        }
    
    def _generate_detection_summary(self, fire_count: int, smoke_count: int, normal_count: int, total: int) -> str:
        """Generate detection status summary"""
        if fire_count > 0 and smoke_count > 0:
            return f"{fire_count}/{total} cameras detecting fire, {smoke_count}/{total} detecting smoke"
        elif fire_count > 0:
            return f"{fire_count}/{total} cameras detecting fire"
        elif smoke_count > 0:
            return f"{smoke_count}/{total} cameras detecting smoke"
        else:
            return f"All {total} cameras normal"
    
    def _parse_nearby_stations(self, station_strings: List[str]) -> List[Dict]:
        """
        Parse nearby_stations string list into structured stations_detail list
        """
        structured = []
        pattern = r"^(.*?)\s*\(([\d.]+)km\s*/\s*([\d.]+)mi\)\s*-\s*(.+)$"
        
        for s in station_strings:
            match = re.match(pattern, s)
            if match:
                name, km, mi, location = match.groups()
                structured.append({
                    "name": name.strip(),
                    "distance_km": float(km),
                    "distance_mi": float(mi),
                    "location": location.strip(),
                    "phone": None
                })
            else:
                structured.append({
                    "name": s[:30],
                    "distance_km": None,
                    "distance_mi": None,
                    "location": s,
                    "phone": None
                })
        return structured
    
    def load_static_geo_data(self, filename: str = "output/nested_california_fire_risk_enhanced_dataset.jsonl") -> Optional[List[Dict]]:
        """
        Load static geographic data
        
        Args:
            filename: Geographic data filename
            
        Returns:
            Geographic data list
        """
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            if not os.path.exists(file_path):
                logger.error(f"Static geographic data file does not exist: {file_path}")
                return None
            
            geo_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        geo_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON on line {line_num} of geographic data: {e}")
                        continue
            
            logger.info(f"Successfully loaded {len(geo_data)} geographic data records")
            self.static_geo_data = geo_data
            return geo_data
            
        except Exception as e:
            logger.error(f"Failed to load static geographic data: {str(e)}")
            return None
    
    def match_nearest_grid_point(self, lat: float, lon: float, max_distance_km: float = 5.0) -> Optional[Dict]:
        """
        Match nearest grid point
        
        Args:
            lat, lon: Query point latitude and longitude
            max_distance_km: Maximum matching distance (km)
            
        Returns:
            Nearest geographic data point, return None if beyond max distance
        """
        if not self.static_geo_data:
            logger.error("Static geographic data not loaded")
            return None
        
        min_distance = float('inf')
        nearest_point = None
        
        for geo_point in self.static_geo_data:
            distance = self.haversine_distance(
                lat, lon,
                geo_point['location']['lat'],
                geo_point['location']['lon']
            ) / 1000  # Convert to kilometers
            
            if distance < min_distance:
                min_distance = distance
                nearest_point = geo_point
        
        if min_distance > max_distance_km:
            logger.warning(f"Nearest grid point distance {min_distance:.2f}km exceeds maximum allowed distance {max_distance_km}km")
            return None
        
        logger.debug(f"Matched grid point successfully, distance: {min_distance:.2f}km")
        return nearest_point
    
    def integrate_weather_data(self, lat: float, lon: float) -> Dict:
        """
        Integrate weather data
        
        Args:
            lat, lon: Latitude and longitude
            
        Returns:
            Weather data dictionary
        """
        try:
            weather_data = get_weather(lat, lon)
            logger.debug(f"Successfully retrieved weather data: {lat}, {lon}")
            return weather_data
        except Exception as e:
            logger.error(f"Failed to retrieve weather data: {str(e)}")
            return {
                "temperature": {"celsius": None, "fahrenheit": None},
                "humidity": None,
                "wind_speed": {"mps": None, "mph": None},
                "weather_description": "Unavailable",
                "source": "OpenWeatherMap",
                "error": str(e)
            }
    
    def generate_unified_assessment_data(self, camera_filename: str = "camera_monitoring_dataset.jsonl") -> Optional[List[Dict]]:
        """
        Generate unified evaluation data format
        
        Args:
            camera_filename: Camera data filename
            
        Returns:
            Unified evaluation data list
        """
        logger.info("Start generating unified evaluation data...")
        
        # 1. Load camera data
        camera_data = self.load_camera_jsonl_data(camera_filename)
        if not camera_data:
            logger.error("Camera data loading failed")
            return None
        
        # 2. Load static geographic data
        if not self.static_geo_data:
            geo_data = self.load_static_geo_data()
            if not geo_data:
                logger.error("Static geographic data loading failed")
                return None
        
        # 3.Aggregate camera data
        camera_clusters = self.cluster_cameras_by_distance(camera_data, self.camera_cluster_radius)
        
        # 4. Generate evaluation data for each cluster
        assessment_data = []
        
        for i, cluster in enumerate(camera_clusters):
            logger.debug(f"Processing camera cluster {i+1}/{len(camera_clusters)}")
            
            # Analyze cluster detection status
            cluster_analysis = self.analyze_cluster_detection(cluster)
            
            # Match nearest grid point
            nearest_grid = self.match_nearest_grid_point(
                cluster_analysis['cluster_center']['lat'],
                cluster_analysis['cluster_center']['lon']
            )
            
            if not nearest_grid:
                logger.warning(f"Cluster {i+1} could not be matched to a suitable grid point, skipping")
                continue
            
            resources = nearest_grid.get("resources", {})
            if "nearby_stations" in resources and "stations_detail" not in resources:
                resources["stations_detail"] = self._parse_nearby_stations(resources["nearby_stations"])

            # Get weather data
            weather_data = self.integrate_weather_data(
                cluster_analysis['cluster_center']['lat'],
                cluster_analysis['cluster_center']['lon']
            )
            
            # Generate unified data format
            unified_data = {
                'cluster_id': f"CLUSTER_{i+1:04d}",
                'location': {
                    'coordinates': cluster_analysis['cluster_center'],
                    'display_name': nearest_grid.get('display_name', 'Unknown'),
                    'city': nearest_grid.get('admin', {}).get('city', 'Unknown'),
                    'county': nearest_grid.get('admin', {}).get('county', 'Unknown'),
                    'land_type': nearest_grid.get('land', {}).get('land_type', 'Unknown'),
                    'nearest_grid_point': nearest_grid.get('location', {}).get('name', 'Unknown')
                },
                'fire_risk_data': nearest_grid.get('fire_risk', {}),
                'land_info': nearest_grid.get('land', {}),
                'camera_assessment': cluster_analysis,
                'fire_stations': nearest_grid.get('resources', {}),
                'weather': weather_data
            }
            
            assessment_data.append(unified_data)
        
        logger.info(f"Unified evaluation data generation completed: {len(assessment_data)} valid clusters")
        return assessment_data
    
    def validate_assessment_data(self, assessment_data: List[Dict]) -> List[Dict]:
        """
        Validate and clean evaluation data
        
        Args:
            assessment_data: Original evaluation data list
            
        Returns:
            Validated data list
        """
        valid_data = []
        
        for i, data in enumerate(assessment_data):
            try:
                # Check required fields
                required_fields = ['location', 'camera_assessment', 'weather']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    logger.warning(f"Evaluation data {i+1} is missing required fields: {missing_fields}")
                    continue
                
                # Validate camera data
                camera_data = data.get('camera_assessment', {})
                if camera_data.get('total_cameras', 0) == 0:
                    logger.warning(f"Evaluation data {i+1} has zero cameras")
                    continue
                
                # Validate geographic data
                location = data.get('location', {})
                coordinates = location.get('coordinates', {})
                if not coordinates.get('lat') or not coordinates.get('lon'):
                    logger.warning(f"Evaluation data {i+1} is missing valid coordinates")
                    continue
                
                # Validate weather data
                weather = data.get('weather', {})
                if weather.get('error'):
                    logger.warning(f"Evaluation data {i+1} failed to retrieve weather data: {weather.get('error')}")
                
                valid_data.append(data)
                
            except Exception as e:
                logger.error(f"Error validating evaluation data {i+1}: {str(e)}")
                continue
        
        logger.info(f"Data validation completed, valid records: {len(valid_data)}/{len(assessment_data)}")
        return valid_data
    
    def save_processed_data_for_debug(self, assessment_data: List[Dict], filename: str = None) -> str:
        """
        Save processed data for debugging (optional feature)
        
        Args:
            assessment_data: Evaluation data list
            filename: Output filename
            
        Returns:
            Saved file path
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'processed_data_debug_{timestamp}.json'
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(assessment_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Debug data saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save debug data: {str(e)}")
            return ""
    
    def get_assessment_data(self, camera_filename: str = "camera_monitoring_dataset.jsonl") -> Optional[List[Dict]]:
        """
        Get processed and validated evaluation data (main interface method)
        
        Args:
            camera_filename: Camera data filename
            
        Returns:
            Processed evaluation data list
        """
        # Generate unified evaluation data
        assessment_data = self.generate_unified_assessment_data(camera_filename)
        
        if not assessment_data:
            logger.error("Failed to generate evaluation data")
            return None
        
        # Validate data
        validated_data = self.validate_assessment_data(assessment_data)
        
        if not validated_data:
            logger.error("No evaluation data passed validation")
            return None
        
        logger.info(f"Evaluation data preparation completed: {len(validated_data)} valid data points")
        return validated_data