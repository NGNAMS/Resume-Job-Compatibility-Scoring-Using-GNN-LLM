"""
Graph builder for resume-job compatibility.
Creates heterogeneous graphs with resume, job, and skill nodes.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId

from ..config import get_config
from ..utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build heterogeneous graphs for resume-job pairs."""
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        similarity_threshold: float = None
    ):
        """
        Initialize graph builder.
        
        Args:
            db_manager: Database manager instance
            similarity_threshold: Cosine similarity threshold for common skills
        """
        config = get_config()
        self.db = db_manager or DatabaseManager()
        self.similarity_threshold = similarity_threshold or config.model.cosine_similarity_threshold
        
        logger.info(f"Initialized GraphBuilder with similarity threshold: {self.similarity_threshold}")
    
    def build_graph(
        self,
        job_id: str,
        resume_id: str,
        include_experience: bool = False,
        include_education: bool = False
    ) -> HeteroData:
        """
        Build a heterogeneous graph for a resume-job pair.
        
        Args:
            job_id: MongoDB ObjectId of the job
            resume_id: MongoDB ObjectId of the resume
            include_experience: Whether to include experience nodes
            include_education: Whether to include education nodes
            
        Returns:
            HeteroData graph object
        """
        # Fetch job and resume data
        job = self.db.get_collection("jobs").find_one({"_id": ObjectId(job_id)})
        resume = self.db.get_collection("resumes").find_one({"_id": ObjectId(resume_id)})
        
        if not job or not resume:
            raise ValueError(f"Job or resume not found: job_id={job_id}, resume_id={resume_id}")
        
        # Extract and match components
        common_skills, job_uncommon_skills, resume_uncommon_skills = self._match_skills(job, resume)
        
        # Build graph
        data = self._construct_hetero_data(
            job, resume,
            common_skills, job_uncommon_skills, resume_uncommon_skills
        )
        
        logger.debug(
            f"Built graph: job_id={job_id}, resume_id={resume_id}, "
            f"common_skills={len(common_skills)}, "
            f"job_uncommon={len(job_uncommon_skills)}, "
            f"resume_uncommon={len(resume_uncommon_skills)}"
        )
        
        return data
    
    def _match_skills(
        self,
        job: Dict,
        resume: Dict
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Match skills between job and resume using cosine similarity.
        
        Args:
            job: Job document
            resume: Resume document
            
        Returns:
            Tuple of (common_skills, job_uncommon_skills, resume_uncommon_skills)
        """
        common_skills = []
        
        for job_skill in job.get("new_skill_set", []):
            best_match = {"org_obj": job_skill, "similarity": 0, "tar_obj": None}
            
            for resume_skill in resume.get("new_skill_set", []):
                try:
                    job_embedding = np.array([job_skill["sentence_embedding_small"]])
                    resume_embedding = np.array([resume_skill["related_skill_sentence_embedding_small"]])
                    
                    similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]
                    
                    if similarity > best_match["similarity"]:
                        best_match["similarity"] = similarity
                        best_match["tar_obj"] = resume_skill
                        
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error computing similarity: {e}")
                    continue
            
            if best_match["similarity"] > self.similarity_threshold:
                common_skills.append(best_match)
        
        # Find uncommon skills
        common_job_skills = [s["org_obj"]["skill"] for s in common_skills]
        common_resume_skills = [s["tar_obj"]["org_skill_title"] for s in common_skills]
        
        job_uncommon_skills = [
            s for s in job.get("new_skill_set", [])
            if s["skill"] not in common_job_skills
        ]
        
        resume_uncommon_skills = [
            s for s in resume.get("new_skill_set", [])
            if s["org_skill_title"] not in common_resume_skills
        ]
        
        return common_skills, job_uncommon_skills, resume_uncommon_skills
    
    def _construct_hetero_data(
        self,
        job: Dict,
        resume: Dict,
        common_skills: List[Dict],
        job_uncommon_skills: List[Dict],
        resume_uncommon_skills: List[Dict]
    ) -> HeteroData:
        """
        Construct HeteroData object from matched skills.
        
        Args:
            job: Job document
            resume: Resume document
            common_skills: List of common skill matches
            job_uncommon_skills: Job-specific skills
            resume_uncommon_skills: Resume-specific skills
            
        Returns:
            HeteroData graph
        """
        data = HeteroData()
        
        # Initialize resume and job nodes with placeholder embeddings
        # These will be updated during GNN message passing
        embedding_dim = 1536  # Or get from config
        data['resume'].x = torch.ones(1, embedding_dim, dtype=torch.float)
        data['job'].x = torch.ones(1, embedding_dim, dtype=torch.float)
        
        # Collect skill embeddings and titles
        job_uncommon_embeddings = [
            s["sentence_embedding_small"] for s in job_uncommon_skills
        ]
        job_uncommon_titles = [s["skill"] for s in job_uncommon_skills]
        
        common_embeddings = [
            s["org_obj"]["sentence_embedding_small"] for s in common_skills
        ]
        common_titles = [s["org_obj"]["skill"] for s in common_skills]
        
        resume_uncommon_embeddings = [
            s["related_skill_sentence_embedding_small"] for s in resume_uncommon_skills
        ]
        resume_uncommon_titles = [s["org_skill_title"] for s in resume_uncommon_skills]
        
        # Combine all skills: job_uncommon + common + resume_uncommon
        all_embeddings = job_uncommon_embeddings + common_embeddings + resume_uncommon_embeddings
        all_titles = job_uncommon_titles + common_titles + resume_uncommon_titles
        
        if len(all_embeddings) == 0:
            logger.warning("No skills found, creating empty skill node")
            all_embeddings = [np.zeros(embedding_dim)]
            all_titles = ["no_skill"]
        
        # Create skill nodes
        data['skill'].x = torch.tensor(all_embeddings, dtype=torch.float)
        data['skill'].title = all_titles
        
        # Create edges: job -> skills
        num_job_connected = len(job_uncommon_skills) + len(common_skills)
        if num_job_connected > 0:
            job_edge_sources = list(range(num_job_connected))
            job_edge_targets = [0] * num_job_connected
            data['skill', 'to', 'job'].edge_index = torch.tensor(
                [job_edge_sources, job_edge_targets],
                dtype=torch.long
            )
        
        # Create edges: resume -> skills
        num_resume_connected = len(resume_uncommon_skills) + len(common_skills)
        if num_resume_connected > 0:
            resume_edge_sources = list(range(
                len(job_uncommon_skills),
                len(job_uncommon_skills) + num_resume_connected
            ))
            resume_edge_targets = [0] * num_resume_connected
            data['skill', 'to', 'resume'].edge_index = torch.tensor(
                [resume_edge_sources, resume_edge_targets],
                dtype=torch.long
            )
        
        return data
    
    def build_dataset(
        self,
        relevance_query: Optional[Dict] = None,
        limit: Optional[int] = None,
        show_progress: bool = True
    ) -> List[HeteroData]:
        """
        Build a dataset of graphs from relevance collection.
        
        Args:
            relevance_query: MongoDB query to filter relevance documents
            limit: Maximum number of graphs to build
            show_progress: Whether to show progress bar
            
        Returns:
            List of HeteroData graphs with compatibility scores
        """
        relevances = self.db.get_collection("relevance").find(
            relevance_query or {},
            limit=limit
        )
        relevances = list(relevances)
        
        logger.info(f"Building dataset from {len(relevances)} job applications...")
        
        dataset = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(relevances, desc="Building graphs")
        else:
            iterator = relevances
        
        for relevance in iterator:
            try:
                data = self.build_graph(
                    job_id=str(relevance["job_id"]),
                    resume_id=str(relevance["resume_id"])
                )
                
                # Add graph ID and compatibility score
                data.graph_id = str(relevance['_id'])
                data.y = torch.tensor([relevance["score"]], dtype=torch.float)
                
                dataset.append(data)
                
            except Exception as e:
                logger.warning(f"Failed to build graph for relevance {relevance.get('_id')}: {e}")
                continue
        
        logger.info(f"Successfully built {len(dataset)} graphs")
        return dataset

