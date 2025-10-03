"""
Database management utilities.
Secure MongoDB connection handling.
"""

import logging
from typing import Optional, Dict, Any, List
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from ..config import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """MongoDB database manager with secure connection handling."""
    
    def __init__(self, uri: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            uri: MongoDB connection URI (if None, loads from config)
            database: Database name (if None, loads from config)
        """
        config = get_config()
        self.uri = uri or config.mongodb.uri
        self.database_name = database or config.mongodb.database
        
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        
        logger.info(f"Initialized DatabaseManager for database: {self.database_name}")
    
    @property
    def client(self) -> MongoClient:
        """Get MongoDB client (lazy initialization)."""
        if self._client is None:
            try:
                self._client = MongoClient(self.uri)
                # Test connection
                self._client.server_info()
                logger.info("Successfully connected to MongoDB")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
        return self._client
    
    @property
    def db(self) -> Database:
        """Get database instance."""
        if self._db is None:
            self._db = self.client[self.database_name]
        return self._db
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection from the database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection object
        """
        return self.db[collection_name]
    
    def find_documents(
        self,
        collection_name: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            projection: Fields to include/exclude
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query, projection)
        
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter (default: count all)
            
        Returns:
            Number of documents
        """
        collection = self.get_collection(collection_name)
        return collection.count_documents(query or {})
    
    def close(self):
        """Close database connection."""
        if self._client:
            self._client.close()
            logger.info("Database connection closed")
            self._client = None
            self._db = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

