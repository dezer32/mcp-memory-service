# MCP Memory Service - Architecture Documentation

## Project Overview

The **MCP Memory Service** is a sophisticated semantic memory and persistent storage service designed for Claude Desktop using the Model Context Protocol (MCP). It provides long-term memory capabilities with semantic search, time-based recall, and tag-based organization, enabling Claude to maintain context across conversations and instances.

### Key Features
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Time-Based Recall**: Natural language time expressions (e.g., "last week", "yesterday morning")
- **Tag-Based Organization**: Flexible tagging system with multiple deletion strategies
- **Cross-Platform Compatibility**: Apple Silicon, Intel, Windows, Linux support
- **Hardware-Aware Optimization**: Automatic detection and optimization for MPS, CUDA, ROCm, DirectML
- **Performance Optimization**: Model caching, query caching, batch operations
- **Dashboard Integration**: Web UI compatibility with JSON APIs

## Architecture

The system follows a **layered architecture** pattern with clear separation of concerns:

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Layer                        │
│  • Protocol handling                                       │
│  • Tool registration and execution                         │
│  • Request/response management                             │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     API Layer                              │
│  • 25+ MCP tools/endpoints                                 │
│  • Input validation and error handling                     │
│  • Response formatting                                     │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                            │
│  • Business logic                                          │
│  • Data transformation                                     │
│  • Performance optimizations                               │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                              │
│  • ChromaDB vector storage                                 │
│  • Embedding generation                                    │
│  • Storage abstractions                                    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Configuration Layer                        │
│  • Path management and validation                          │
│  • Platform-specific settings                              │
│  • Environment configuration                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Model

#### Memory Entity
```python
@dataclass
class Memory:
    content: str                          # The actual memory content
    content_hash: str                     # SHA-256 hash for deduplication
    tags: List[str]                       # Categorization tags
    memory_type: Optional[str]            # Type classification
    metadata: Dict[str, Any]              # Additional metadata
    embedding: Optional[List[float]]      # Vector embedding
    
    # Sophisticated timestamp handling
    created_at: Optional[float]           # Unix timestamp (float)
    created_at_iso: Optional[str]         # ISO 8601 string
    updated_at: Optional[float]           # Unix timestamp (float)  
    updated_at_iso: Optional[str]         # ISO 8601 string
    timestamp: datetime                   # Legacy datetime field
```

#### Query Result Entity
```python
@dataclass
class MemoryQueryResult:
    memory: Memory                        # The retrieved memory
    relevance_score: float                # Cosine similarity score
    debug_info: Dict[str, Any]           # Debug information
```

## API Endpoints

The service exposes 25+ MCP tools organized into functional categories:

### Core Memory Operations
| Tool | Description | Location |
|------|-------------|----------|
| `store_memory` | Store new information with optional tags | server.py:892 |
| `retrieve_memory` | Semantic search for relevant memories | server.py:950 |
| `recall_memory` | Time-based recall with natural language | server.py:1156 |
| `search_by_tag` | Find memories by tags (array support) | server.py:1004 |
| `exact_match_retrieve` | Find memories with exact content match | server.py:1138 |
| `debug_retrieve` | Retrieve with similarity scores | server.py:1110 |

### Memory Management  
| Tool | Description | Location |
|------|-------------|----------|
| `delete_memory` | Delete specific memory by hash | server.py:1039 |
| `delete_by_tag` | Enhanced tag deletion (single/multiple) | server.py:1050 |
| `delete_by_tags` | Explicit multi-tag deletion (OR logic) | server.py:1067 |
| `delete_by_all_tags` | Delete with ALL specified tags (AND logic) | server.py:1075 |
| `cleanup_duplicates` | Remove duplicate entries | server.py:1083 |

### Time-Based Operations
| Tool | Description | Location |
|------|-------------|----------|
| `recall_by_timeframe` | Retrieve memories within date range | server.py:1377 |
| `delete_by_timeframe` | Delete memories within date range | server.py:1419 |
| `delete_before_date` | Delete memories before specific date | server.py:1441 |

### Database Operations
| Tool | Description | Location |
|------|-------------|----------|
| `check_database_health` | Health check with performance metrics | server.py:1315 |
| `get_embedding` | Get raw embedding vector | server.py:1091 |
| `check_embedding_model` | Verify model status | server.py:1102 |

### Dashboard Operations (JSON APIs)
| Tool | Description | Location |
|------|-------------|----------|
| `dashboard_check_health` | JSON health status | server.py:714 |
| `dashboard_recall_memory` | JSON recall results | server.py:732 |
| `dashboard_retrieve_memory` | JSON search results | server.py:799 |
| `dashboard_search_by_tag` | JSON tag search | server.py:837 |
| `dashboard_get_stats` | JSON database statistics | server.py:875 |
| `dashboard_optimize_db` | JSON optimization results | server.py:974 |
| `dashboard_create_backup` | JSON backup creation | server.py:993 |
| `dashboard_delete_memory` | JSON memory deletion | server.py:1039 |

## Technology Stack

### Core Technologies
- **Language**: Python 3.10+ with type hints and dataclasses
- **Protocol**: Model Context Protocol (MCP) 2024-11-05
- **Vector Database**: ChromaDB 0.5.23 with DuckDB+Parquet backend
- **Embeddings**: Sentence Transformers with hardware acceleration
- **Async Framework**: asyncio for non-blocking I/O operations

### Hardware Acceleration Support
| Platform | Accelerator | Status | Configuration |
|----------|-------------|---------|---------------|
| macOS Apple Silicon | MPS | ✅ Full Support | PYTORCH_ENABLE_MPS_FALLBACK=1 |
| macOS Intel | CPU | ✅ Full Support | Optimized batch sizes |
| Windows | CUDA | ✅ Full Support | CUDA toolkit auto-detection |
| Windows | DirectML | ✅ Supported | torch-directml package |
| Linux | CUDA | ✅ Full Support | CUDA toolkit auto-detection |
| Linux | ROCm | ✅ Supported | ROCm installation detection |
| All Platforms | CPU | ✅ Fallback | ONNX Runtime option |

### Dependencies
- **chromadb==0.5.23**: Vector database with HNSW indexing
- **sentence-transformers**: Embedding model management  
- **tokenizers==0.20.3**: Text tokenization
- **mcp>=1.0.0,<2.0.0**: Model Context Protocol implementation
- **torch**: PyTorch framework (platform-specific versions)

## Project Structure

```
mcp-memory-service/
├── src/mcp_memory_service/           # Core package
│   ├── __init__.py                   # Package initialization
│   ├── server.py                     # Main MCP server (1400+ lines)
│   ├── config.py                     # Configuration and path management  
│   ├── models/                       # Data models
│   │   ├── __init__.py
│   │   └── memory.py                 # Memory and MemoryQueryResult classes
│   ├── storage/                      # Storage layer
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract storage interface
│   │   ├── chroma.py                 # ChromaDB implementation (1000+ lines)
│   │   └── chroma_enhanced.py        # Performance optimizations
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── time_parser.py            # Natural language time parsing
│       ├── hashing.py                # Content hash generation
│       ├── system_detection.py       # Hardware detection and optimization
│       ├── db_utils.py               # Database utilities and validation
│       ├── debug.py                  # Debug and diagnostic tools
│       └── utils.py                  # General utilities
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── performance/                  # Performance tests
│   ├── test_memory_ops.py            # Memory operations testing
│   ├── test_semantic_search.py       # Search functionality testing
│   ├── test_tag_storage.py           # Tag system testing
│   ├── test_time_parser.py           # Time parsing testing
│   ├── test_timestamp_recall.py      # Timestamp recall testing
│   └── test_database.py              # Database testing
├── scripts/                          # Development and maintenance scripts
│   ├── install.py                    # Cross-platform installation
│   ├── install_windows.py            # Windows-specific installation
│   ├── test_installation.py          # Installation verification
│   ├── verify_environment.py         # Environment validation
│   ├── migrations/                   # Database migration scripts
│   ├── backup_memories.py            # Backup utilities
│   ├── repair_memories.py            # Repair utilities
│   └── validate_memories.py          # Validation utilities
├── docs/                            # Documentation
│   ├── guides/                      # User guides
│   ├── examples/                    # Usage examples
│   ├── technical/                   # Technical documentation
│   └── implementation/              # Implementation details
├── memory_wrapper.py                # Windows wrapper script
├── install.py                       # Main installation script
├── pyproject.toml                   # Project configuration
├── requirements.txt                 # Python dependencies
├── pytest.ini                      # Test configuration
├── setup.py                        # Setup script with platform detection
└── README.md                       # Project documentation
```

## Testing Strategy

### Test Organization
The project uses **pytest** with a structured approach to testing:

#### Test Categories
1. **Unit Tests** (`tests/unit/`)
   - Individual component testing
   - Mock dependencies for isolation
   - Fast execution (<1s each)

2. **Integration Tests** (`tests/integration/`)
   - End-to-end workflow testing
   - Real database interactions
   - Cross-component validation

3. **Performance Tests** (`tests/performance/`)
   - Query performance benchmarking
   - Memory usage validation
   - Caching effectiveness measurement

#### Key Test Files
- `test_memory_ops.py`: Memory CRUD operations
- `test_semantic_search.py`: Vector search functionality
- `test_tag_storage.py`: Tag system operations
- `test_time_parser.py`: Natural language time parsing
- `test_timestamp_recall.py`: Time-based memory recall
- `test_database.py`: Database health and validation

#### Test Configuration
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: unit tests
    integration: integration tests  
    performance: performance tests
    asyncio: mark test as async
```

### Testing Commands
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance

# Run specific test files
pytest tests/test_memory_ops.py
pytest tests/test_semantic_search.py

# Run with coverage
pytest --cov=mcp_memory_service tests/
```

## Development Commands

### Installation and Setup
```bash
# Cross-platform installation with auto-detection
python install.py

# Development mode installation  
python install.py --dev

# Windows-specific installation
python scripts/install_windows.py

# Force compatible dependencies (macOS Intel)
python install.py --force-compatible-deps
```

### Package Management
```bash
# Using pip (traditional)
pip install -e .

# Using uv (modern, faster)
uv pip install -e .

# Install with specific PyTorch version
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Testing and Validation
```bash
# Comprehensive installation test
python scripts/test_installation.py

# Environment verification
python scripts/verify_environment.py

# PyTorch Windows verification
python scripts/verify_pytorch_windows.py

# Database validation
python scripts/validate_memories.py
```

### Database Management
```bash
# Create backup
python scripts/backup_memories.py

# Repair database
python scripts/repair_memories.py

# Migrate timestamps
python scripts/migrate_timestamps.py

# Clean up duplicates
python scripts/cleanup_memories.py
```

### Development Tools
```bash
# Run memory server directly
python scripts/run_memory_server.py

# Debug dependencies
python scripts/debug_dependencies.py

# Fix sitecustomize issues
python scripts/fix_sitecustomize.py

# Convert to uv package manager
python scripts/convert_to_uv.py
```

## Environment Setup

### Hardware-Aware Installation
The installation process automatically detects system capabilities:

1. **System Detection**
   - Operating system and architecture
   - Available memory and CPU cores
   - Python version and virtual environment

2. **GPU Detection**
   - CUDA availability and version
   - ROCm support (Linux)
   - Apple Metal Performance Shaders (macOS)
   - DirectML support (Windows)

3. **Optimization Selection**
   - Optimal embedding model selection
   - Batch size tuning based on memory
   - Device selection (GPU/CPU)
   - Model caching strategies

### Configuration Files

#### Environment Variables
```bash
# Required paths
MCP_MEMORY_CHROMA_PATH=/path/to/chroma_db
MCP_MEMORY_BACKUPS_PATH=/path/to/backups

# Optional optimizations
MCP_MEMORY_MODEL_NAME=all-mpnet-base-v2
MCP_MEMORY_BATCH_SIZE=32
MCP_MEMORY_USE_ONNX=0
MCP_MEMORY_USE_DIRECTML=0

# Hardware-specific
PYTORCH_ENABLE_MPS_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-memory-service",
        "run",
        "memory"
      ],
      "env": {
        "MCP_MEMORY_CHROMA_PATH": "/path/to/chroma_db",
        "MCP_MEMORY_BACKUPS_PATH": "/path/to/backups"
      }
    }
  }
}
```

### Platform-Specific Setup

#### macOS Apple Silicon
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python install.py
```

#### Windows with CUDA
```bash
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
python scripts/install_windows.py
```

#### Linux with ROCm
```bash
export ROCM_HOME=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python install.py
```

## Development Guidelines

### Code Style and Patterns

#### Language Requirements
- **Python 3.10+** minimum version
- **Type hints** for all function signatures
- **Dataclasses** for data structures
- **Async/await** for I/O operations

#### Code Organization
```python
# File header template
"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License.
"""

# Import organization
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .models.memory import Memory
```

#### Function Documentation
```python
async def store_memory(self, memory: Memory) -> Tuple[bool, str]:
    """Store a memory with optimized performance.
    
    Args:
        memory: Memory object to store
        
    Returns:
        Tuple of (success, message)
        
    Raises:
        RuntimeError: If collection not initialized
    """
```

#### Error Handling
```python
try:
    # Main operation
    result = await storage.operation()
    return True, "Success message"
except SpecificException as e:
    logger.error(f"Specific error: {str(e)}")
    return False, f"Error: {str(e)}"
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    logger.error(traceback.format_exc())
    return False, f"Unexpected error: {str(e)}"
```

### Performance Guidelines

#### Memory Management
- Use global model caching for embedding models
- Implement LRU caching for frequent queries
- Batch operations when possible
- Clear caches during memory pressure

#### Database Optimization
```python
# Optimized HNSW settings for ChromaDB
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100, 
    "hnsw:M": 16,
    "hnsw:max_elements": 100000
}
```

#### Async Best Practices
```python
# Use async/await for I/O operations
async def async_operation():
    results = await storage.retrieve(query)
    return results

# Batch concurrent operations
async def batch_operations():
    tasks = [operation(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

### Framework Integration

#### MCP Protocol Compliance
- Implement all required MCP methods
- Use proper MCP types for requests/responses
- Handle protocol errors gracefully
- Support multiple protocol versions

#### Testing Requirements
- Write tests before implementing features
- Achieve >80% code coverage
- Test both success and failure paths
- Include performance regression tests

## Security Considerations

### Data Protection
- **No Secrets in Logs**: Never log sensitive information
- **Path Validation**: Validate all file paths before use
- **Input Sanitization**: Sanitize all user inputs
- **Content Hashing**: Use SHA-256 for content integrity

### File System Security
```python
def validate_and_create_path(path: str) -> str:
    """Validate and create directory path with security checks."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    
    # Prevent path traversal
    if '..' in abs_path or not abs_path.startswith('/'):
        raise SecurityError("Invalid path")
    
    # Test write permissions safely
    os.makedirs(abs_path, exist_ok=True)
    test_file = os.path.join(abs_path, '.write_test')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    
    return abs_path
```

### Error Handling Security
- **Information Disclosure**: Sanitize error messages
- **Exception Safety**: Never expose internal state
- **Graceful Degradation**: Fail safely without data loss
- **Audit Logging**: Log security-relevant operations

### Memory Security
- **Sensitive Data**: Clear sensitive data from memory
- **Model Caching**: Secure model file storage
- **Temporary Files**: Clean up temporary files
- **Resource Limits**: Implement resource usage limits

## Future Considerations

### Performance Enhancements
1. **Advanced Caching**
   - Redis integration for distributed caching
   - Persistent query result caching
   - Embedding compression techniques

2. **Scalability Improvements**
   - Horizontal database sharding
   - Load balancing for multiple instances
   - Distributed embedding computation

3. **Hardware Optimization**
   - Quantized model support (INT8/FP16)
   - Custom CUDA kernels for search
   - Edge deployment optimizations

### Feature Enhancements
1. **Advanced Search**
   - Hybrid semantic + keyword search
   - Multi-modal embedding support (text + images)
   - Temporal relevance weighting

2. **Enhanced Time Parsing**
   - Support for more natural language patterns
   - Timezone-aware time handling
   - Relative time expressions (e.g., "2 meetings ago")

3. **Collaboration Features**
   - Shared memory spaces
   - Permission-based access control
   - Memory synchronization across devices

### Integration Possibilities
1. **External Services**
   - Vector database alternatives (Pinecone, Weaviate)
   - Cloud storage backends (S3, GCS)
   - Monitoring and analytics platforms

2. **Development Tools**
   - Visual Studio Code extension
   - Jupyter notebook integration
   - Command-line interface improvements

3. **API Extensions**
   - REST API for web applications
   - GraphQL support for flexible queries
   - WebSocket support for real-time updates

### Architecture Evolution
1. **Microservices Architecture**
   - Separate embedding service
   - Dedicated search service
   - Independent configuration service

2. **Plugin System**
   - Custom embedding model plugins
   - Storage backend plugins
   - Search algorithm plugins

3. **Cloud-Native Features**
   - Kubernetes deployment support
   - Container optimization
   - Health check endpoints

---

*This architecture documentation provides a comprehensive overview of the MCP Memory Service codebase. For implementation details, refer to the source code and additional documentation in the `docs/` directory.*