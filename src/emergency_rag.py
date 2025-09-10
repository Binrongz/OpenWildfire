#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emergency RAG
Use RAG techniques to augment AI-generated emergency recommendations based on professional documents
"""

import json
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmergencyRAG:
    """Emergency RAG System"""
    
    def __init__(self, docs_dir: str = None):
        """
        Init RAG
        
        Args:
            docs_dir: Document directory path
        """
        if docs_dir is None:
            # path：src/data/emergency_docs/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            docs_dir = os.path.join(os.path.dirname(current_dir), 'data', 'emergency_docs')
        
        self.docs_dir = Path(docs_dir)
        self.pdfs_dir = self.docs_dir / 'pdfs'
        self.vectorstore_dir = self.docs_dir / 'vectorstore'
        self.config_file = self.docs_dir / 'config.json'
        
        # Ensure directory exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(exist_ok=True)
        self.vectorstore_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        self.vectorstore = None
        self.text_splitter = None
        self.embeddings = None
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_components()
        else:
            logger.warning("RAG system not fully initialized: LangChain unavailable")
    
    def _load_config(self) -> Dict:
        """Load configuration file"""
        default_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k_results": 8,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "pdf_auto_reload": False,
            "max_recommendations": 12
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge default configuration
                default_config.update(config)
                logger.info("Configuration file loaded successfully")
            except Exception as e:
                logger.warning(f"Configuration file loading failed, using default configuration: {e}")
        else:
            # Create default configuration file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info("Create default configuration file")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config['embedding_model']
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            
            self._initialize_vectorstore()
            
            logger.info("RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"RAG components initialization failed: {e}")
            self.vectorstore = None
    
    def _initialize_vectorstore(self):
        """Initialize vector store"""
        try:
            if (self.vectorstore_dir / 'chroma.sqlite3').exists():
                self.vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_dir),
                    embedding_function=self.embeddings
                )
                logger.info("Load existing vector store")
            else:
                self._build_vectorstore()
        
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            self.vectorstore = None
    
    def _build_vectorstore(self):
        """Build vector store"""
        pdf_files = list(self.pdfs_dir.glob('*.pdf'))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdfs_dir}")
            return
        
        logger.info(f"Start processing {len(pdf_files)} PDF files")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Process file: {pdf_file.name}")
                
                # Load PDF
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Split document
                splits = self.text_splitter.split_documents(documents)
                
                # Add metadata
                for split in splits:
                    split.metadata['source_file'] = pdf_file.name
                    split.metadata['doc_type'] = 'emergency_procedure'
                
                all_documents.extend(splits)
                logger.info(f"{pdf_file.name}: Generate {len(splits)} document chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}")
        
        if all_documents:
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=str(self.vectorstore_dir)
            )
            
            self.vectorstore.persist()
            logger.info(f"Vector store built with {len(all_documents)} document chunks")
        else:
            logger.warning("No documents processed successfully")
    
    def enhance_recommendations(self, original_recs: List[str], assessment_data: Dict) -> List[str]:
        """
        Enhance emergency suggestions
        
        Args:
            original_recs: AI-generated original suggestions
            assessment_data: Wildfire risk assessment data
            
        Returns:
            Enhanced suggestions list
        """
        if not LANGCHAIN_AVAILABLE or not self.vectorstore:
            logger.warning("RAG system unavailable, return original suggestions")
            return original_recs
        
        try:
            # Build query context
            query_context = self._build_query_context(assessment_data)
            
            # Retrieve relevant documents
            relevant_docs = self._retrieve_relevant_docs(query_context)
            
            # Generate enhanced suggestions
            enhanced_recs = self._generate_enhanced_recommendations(
                original_recs, 
                relevant_docs, 
                assessment_data
            )
            
            logger.info(f"Suggestion enhancement completed: {len(original_recs)} -> {len(enhanced_recs)}")
            return enhanced_recs
            
        except Exception as e:
            logger.error(f"Suggestion enhancement failed: {e}")
            return original_recs
    
    def _build_query_context(self, assessment_data: Dict) -> str:
        """Build query context"""
        context_parts = []
        
        # Risk level
        if 'risk_level' in assessment_data:
            context_parts.append(f"risk level {assessment_data['risk_level']}")
        
        # Camera detection
        camera_data = assessment_data.get('camera_assessment', {})
        if camera_data.get('cameras_detecting_fire', 0) > 0:
            context_parts.append("fire detection")
        if camera_data.get('cameras_detecting_smoke', 0) > 0:
            context_parts.append("smoke detection")
        
        # Weather conditions
        weather = assessment_data.get('weather', {})
        if weather.get('humidity'):
            humidity = str(weather['humidity']).replace('%', '')
            if int(humidity) < 25:
                context_parts.append("low humidity")
        
        wind_speed = weather.get('wind_speed', {}).get('mph', 0)
        if wind_speed > 15:
            context_parts.append("high wind")
        
        # Geographic location
        location = assessment_data.get('location', {})
        if 'county' in location:
            context_parts.append(f"california {location['county']}")
        
        return " ".join(context_parts)
    
    def _retrieve_relevant_docs(self, query: str) -> List[str]:
        """Retrieve relevant documents"""
        try:
            results = self.vectorstore.similarity_search(
                query, 
                k=self.config['top_k_results']
            )
            
            relevant_texts = []
            for doc in results:
                # Clean text
                text = doc.page_content.strip()
                if len(text) > 50:  # Filter out too short text
                    relevant_texts.append(text)
            
            logger.info(f"Retrieved {len(relevant_texts)} relevant document fragments")
            return relevant_texts
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _generate_enhanced_recommendations(self, original_recs: List[str], 
                                         relevant_docs: List[str], 
                                         assessment_data: Dict) -> List[str]:
        """Generate enhanced recommendations"""
        enhanced_recs = []
        
        # 1. Generate enhanced suggestions
        cleaned_original = self._clean_recommendations(original_recs)
        enhanced_recs.extend(cleaned_original[:6])
        
        # 2. Extract relevant suggestions from RAG documents
        rag_recommendations = self._extract_rag_recommendations(
            relevant_docs, 
            assessment_data
        )
        
        # 3. Smart merge to avoid duplicates
        final_recs = self._merge_recommendations(enhanced_recs, rag_recommendations)
        
        # 4. Limit total count
        max_recs = self.config['max_recommendations']
        return final_recs[:max_recs]
    
    def _clean_recommendations(self, recs: List[str]) -> List[str]:
        """Clean suggestion text"""
        cleaned = []
        for rec in recs:
            if isinstance(rec, str) and len(rec.strip()) > 20:
                clean_rec = rec.strip()
                # Remove extra punctuation
                if clean_rec.endswith('.'):
                    clean_rec = clean_rec[:-1]
                cleaned.append(clean_rec)
        return cleaned
    
    def _extract_rag_recommendations(self, docs: List[str], assessment_data: Dict) -> List[str]:
        """Extract suggestions from RAG documents"""
        recommendations = []
        
        # Extended firefighting keywords
        fire_keywords = [
            'fire', 'smoke', 'emergency', 'response', 'evacuation', 'safety', 
            'alert', 'warning', 'procedure', 'protocol', 'action', 'measure',
            'deploy', 'activate', 'establish', 'coordinate', 'notify', 'contact',
            'monitor', 'inspect', 'maintain', 'prepare', 'secure', 'protect'
        ]
        
        # Suggestion verbs
        action_words = [
            'should', 'must', 'ensure', 'activate', 'deploy', 'establish',
            'implement', 'conduct', 'perform', 'execute', 'initiate', 'commence',
            'coordinate', 'notify', 'alert', 'contact', 'maintain', 'monitor',
            'inspect', 'check', 'verify', 'confirm', 'secure', 'protect'
        ]
        
        for doc in docs:
            # Sentence segmentation
            sentences = doc.replace('\n', ' ').split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    sentence_lower = sentence.lower()
                    
                    # Check if contains firefighting-related keywords
                    has_fire_keyword = any(keyword in sentence_lower for keyword in fire_keywords)
                    
                    # Check if it is a suggestive sentence
                    has_action_word = any(word in sentence_lower for word in action_words)
                    
                    # Or contains typical suggestion patterns
                    is_recommendation = (
                        sentence.strip().endswith(':') or
                        'recommend' in sentence_lower or
                        'suggest' in sentence_lower or
                        'advise' in sentence_lower or
                        sentence_lower.startswith('•') or
                        sentence_lower.startswith('-') or
                        sentence_lower.startswith('*')
                    )
                    
                    if has_fire_keyword and (has_action_word or is_recommendation):
                        # Clean sentences
                        clean_sentence = sentence.strip()
                        if clean_sentence and len(clean_sentence) > 15:
                            recommendations.append(clean_sentence)
        
        # Return more suggestions
        return recommendations[:8]
    
    def _merge_recommendations(self, original: List[str], rag_recs: List[str]) -> List[str]:
        """Merge suggestions, avoid duplicates"""
        merged = original.copy()
        
        for rag_rec in rag_recs:
            # Check if duplicate with existing suggestions
            is_duplicate = False
            for existing in merged:
                # Similarity check
                if self._are_similar(rag_rec, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(rag_rec)
        
        return merged
    
    def _are_similar(self, text1: str, text2: str, threshold: float = 0.4) -> bool:
        """Check if two texts are similar"""
        # Simple word overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Remove common stopwords to improve accuracy
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > threshold
    
    def add_pdf_document(self, pdf_path: str) -> bool:
        """Add new PDF document"""
        try:
            if not LANGCHAIN_AVAILABLE or not self.vectorstore:
                logger.error("RAG system unavailable")
                return False
            
            # Copy PDF to docs directory
            import shutil
            pdf_name = os.path.basename(pdf_path)
            target_path = self.pdfs_dir / pdf_name
            shutil.copy2(pdf_path, target_path)
            
            # Rebuild vector store
            self._build_vectorstore()
            
            logger.info(f"PDF document added successfully: {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"Add PDF document failed: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get RAG system status"""
        pdf_files = list(self.pdfs_dir.glob('*.pdf')) if self.pdfs_dir.exists() else []
        
        status = {
            'langchain_available': LANGCHAIN_AVAILABLE,
            'vectorstore_ready': self.vectorstore is not None,
            'pdf_count': len(pdf_files),
            'pdf_files': [f.name for f in pdf_files],
            'config': self.config
        }
        
        return status