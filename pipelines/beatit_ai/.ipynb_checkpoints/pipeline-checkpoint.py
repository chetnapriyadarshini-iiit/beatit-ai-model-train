"""Example workflow pipeline script for abalone pipeline.
                                                                                 . -ModelStep
                                                                                .
    Process-> DataQualityCheck/DataBiasCheck -> Train -> Evaluate -> Condition .
                                                  |                              .
                                                  |                                . -(stop)
                                                  |
                                                   -> CreateModel-> ModelBiasCheck/ModelExplainabilityCheck
                                                           |
                                                           |
                                                            -> BatchTransform -> ModelQualityCheck

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import (
    ConditionLessThanOrEqualTo,
    ConditionGreaterThanOrEqualTo,
)

from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="ChurnPackageGroup",
    pipeline_name="ChurnPipeline",
    base_job_prefix="churn",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    sagemaker_project_name=None,
):
    """Get a SageMaker ML Pipeline for churn training/eval/monitoring."""
    sagemaker_session = get_session(region, default_bucket)
    default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # pipeline parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    input_data = ParameterString(name="InputDataUrl", default_value=f"s3://beatit-ai-data/raw/")

    # data quality/bias/model toggles
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=False)
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value="")
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value="")

    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=False)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value="")

    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value="")
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value="")

    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=False)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value="")

    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=False)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=False)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value="")

    # ---- preprocessing (SKLearn Processor) ----
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-churn-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/raw"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=[
            "--raw-data-dir", "/opt/ml/processing/raw",
            "--train-output-dir", "/opt/ml/processing/train",
            "--val-output-dir", "/opt/ml/processing/validation",
            "--test-output-dir", "/opt/ml/processing/test",
        ],
    )

    step_process = ProcessingStep(name="PreprocessChurnData", step_args=step_args)

    # ---- CheckJobConfig (shared) ----
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        sagemaker_session=pipeline_session,
    )

    # ---- Data Quality Check ----
    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'dataqualitycheckstep']),
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name,
    )

    # ---- Data Bias (Clarify) ----
    data_bias_analysis_cfg_output_path = f"s3://{default_bucket}/{base_job_prefix}/databiascheckstep/analysis_cfg"

    data_bias_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'databiascheckstep']),
        label=0,
        dataset_type="text/csv",
        s3_analysis_config_output_path=data_bias_analysis_cfg_output_path,
    )

    data_bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name=[1],  # registered_via at column index 1 (ensure preprocess reorders)
    )

    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=data_bias_config,
    )

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_data_bias,
        register_new_baseline=register_new_baseline_data_bias,
        model_package_group_name=model_package_group_name,
    )

    # ---- Training (built-in XGBoost) ----
    model_path = f"s3://{default_bucket}/{base_job_prefix}/ChurnTrain"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/churn-train",
        sagemaker_session=pipeline_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        num_round=200,
        max_depth=6,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
    )

    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    step_train = TrainingStep(
        name="TrainChurnModel",
        step_args=step_args,
        depends_on=["DataQualityCheckStep", "DataBiasCheckStep"],
    )

    # ---- Create Model Step (for batch transform / register) ----
    model_for_create = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    create_model_args = model_for_create.create(instance_type="ml.m5.large", accelerator_type="ml.eia1.medium")

    step_create_model = ModelStep(name="ChurnCreateModel", step_args=create_model_args)

    # ---- Batch Transform (to produce predictions + labels for quality checks) ----
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}/{base_job_prefix}/transform-output",
        sagemaker_session=pipeline_session,
    )

    transform_inputs = TransformInput(data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri)

    step_args = transformer.transform(
        data=transform_inputs.data,
        input_filter="$[1:]",   # send features only (label is at col 0)
        join_source="Input",
        output_filter="$[0,-1]",  # predicted, original label
        content_type="text/csv",
        split_type="Line",
    )

    step_transform = TransformStep(name="ChurnTransform", step_args=step_args)

    # ---- Model Quality Check (Binary Classification) ----
    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelqualitycheckstep']),
        problem_type='BinaryClassification',
        inference_attribute='_c0',
        ground_truth_attribute='_c1',
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name,
    )

    # ---- Model Bias Check (Clarify) ----
    model_bias_analysis_cfg_output_path = f"s3://{default_bucket}/{base_job_prefix}/modelbiascheckstep/analysis_cfg"

    model_bias_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelbiascheckstep']),
        s3_analysis_config_output_path=model_bias_analysis_cfg_output_path,
        label=0,
        dataset_type="text/csv",
    )

    model_config = ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type='ml.m5.large',
    )

    model_bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name=[1],
    )

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=model_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig(),
    )

    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_bias,
        register_new_baseline=register_new_baseline_model_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
        model_package_group_name=model_package_group_name,
    )

    # ---- Model Explainability (SHAP) ----
    model_explainability_analysis_cfg_output_path = f"s3://{default_bucket}/{base_job_prefix}/modelexplainabilitycheckstep/analysis_cfg"

    model_explainability_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelexplainabilitycheckstep']),
        s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
        label=0,
        dataset_type="text/csv",
    )

    shap_config = SHAPConfig(seed=123, num_samples=10)

    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )

    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_explainability,
        register_new_baseline=register_new_baseline_model_explainability,
        supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
        model_package_group_name=model_package_group_name,
    )

    # ---- Evaluation Step (script evaluate.py) ----
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-churn-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = script_eval.run(
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )

    evaluation_report = PropertyFile(name="ChurnEvaluationReport", output_name="evaluation", path="evaluation.json")

    step_eval = ProcessingStep(name="EvaluateChurnModel", step_args=step_args, property_files=[evaluation_report])

    # ---- Model registration (ModelPackage) ----
    model_for_register = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    register_step_args = model_for_register.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=ModelMetrics(  # minimal placeholder - replace with your collected metrics
            model_data_statistics=MetricsSource(s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics, content_type="application/json"),
            model_data_constraints=MetricsSource(s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
        ),
        drift_check_baselines=DriftCheckBaselines(
            model_data_statistics=MetricsSource(s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics, content_type="application/json"),
            model_data_constraints=MetricsSource(s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints, content_type="application/json"),
        ),
    )

    step_register = ModelStep(name="RegisterChurnModel", step_args=register_step_args)

    # ---- Condition: gate registration on AUC ----
    cond_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=step_eval.name, property_file=evaluation_report, json_path="binary_classification_metrics.auc.value"),
        right=0.80,
    )

    step_cond = ConditionStep(name="CheckAUCChurnEvaluation", conditions=[cond_auc], if_steps=[step_register], else_steps=[])

    # ---- Pipeline instance ----
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            skip_check_data_quality,
            register_new_baseline_data_quality,
            supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints_data_quality,
            skip_check_data_bias,
            register_new_baseline_data_bias,
            supplied_baseline_constraints_data_bias,
            skip_check_model_quality,
            register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality,
            supplied_baseline_constraints_model_quality,
            skip_check_model_bias,
            register_new_baseline_model_bias,
            supplied_baseline_constraints_model_bias,
            skip_check_model_explainability,
            register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability,
        ],
        steps=[
            step_process,
            data_quality_check_step,
            data_bias_check_step,
            step_train,
            step_create_model,
            step_transform,
            model_quality_check_step,
            model_bias_check_step,
            model_explainability_check_step,
            step_eval,
            step_cond,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline
