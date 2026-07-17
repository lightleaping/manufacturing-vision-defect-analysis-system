"""객체 탐지 데이터셋 분석 계층.

기존 이미지 분류 파이프라인과 객체 탐지 파이프라인을 분리해
분류 V1의 동작을 보호한다.
"""

from .annotation_parser import (
    AnnotationParseError,
    BoundingBox,
    DetectionObject,
    ParsedAnnotation,
    ValidationIssue,
    parse_pascal_voc_annotation,
    validate_annotation,
)
from .dataset_analysis import (
    DatasetAnalysisResult,
    ImageAnnotationRecord,
    analyze_detection_dataset,
    combine_analysis_results,
    save_analysis_json,
)
from .dataset_config import (
    DETECTION_MODEL_CLASS_TO_INDEX,
    DETECTION_SOURCE_CLASS_TO_INDEX,
    NEU_DET_CANONICAL_CLASSES,
    DetectionDatasetCollectionLayout,
    DetectionDatasetConfig,
    DetectionDatasetLayout,
    DetectionDatasetPartitionLayout,
    SplitRatios,
    build_default_config,
    build_partition_config,
    discover_neu_det_layout,
    discover_neu_det_partitions,
    normalize_annotation_class_name,
    normalize_partition_name,
)
from .dataset_split import (
    SplitManifest,
    build_existing_split_manifest,
    build_source_preserving_split_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)

__all__ = [
    "AnnotationParseError",
    "BoundingBox",
    "DetectionObject",
    "ParsedAnnotation",
    "ValidationIssue",
    "parse_pascal_voc_annotation",
    "validate_annotation",
    "DatasetAnalysisResult",
    "ImageAnnotationRecord",
    "analyze_detection_dataset",
    "combine_analysis_results",
    "save_analysis_json",
    "DETECTION_MODEL_CLASS_TO_INDEX",
    "DETECTION_SOURCE_CLASS_TO_INDEX",
    "NEU_DET_CANONICAL_CLASSES",
    "DetectionDatasetCollectionLayout",
    "DetectionDatasetConfig",
    "DetectionDatasetLayout",
    "DetectionDatasetPartitionLayout",
    "SplitRatios",
    "build_default_config",
    "build_partition_config",
    "discover_neu_det_layout",
    "discover_neu_det_partitions",
    "normalize_annotation_class_name",
    "normalize_partition_name",
    "SplitManifest",
    "build_existing_split_manifest",
    "build_source_preserving_split_manifest",
    "build_split_manifest",
    "load_split_manifest",
    "save_split_manifest",
    "validate_split_manifest",
]
