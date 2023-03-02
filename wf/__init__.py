"""
Latch Workflow for executing DeepVariant https://github.com/google/deepvariant
"""

from latch import workflow, large_gpu_task, small_task
from latch.types import (
    LatchFile,
    LatchDir,
)
from latch.resources.launch_plan import LaunchPlan
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional, List


class ModelType(Enum):
    WGS = "WGS"
    WES = "WES"
    PACBIO = "PACBIO"
    PACBIO_ILLUMINA_HYBRID = "PACBIO_ILLUMINA_HYBRID"


def index_fasta(
    ref: LatchFile,
) -> None:
    _cmd = ["samtools", "faidx", ref.local_path]
    subprocess.run(_cmd)


def sort_reads(reads: LatchFile,
) -> None:
    
    sorted_path = reads.local_path + ".sorted"

    sort_cmd = ["samtools", "sort", reads.local_path, "-o", sorted_path]
    mv_cmd = ["mv", sorted_path, reads.local_path]

    subprocess.run(sort_cmd)
    subprocess.run(mv_cmd)



def index_reads(
    reads: LatchFile,
) -> None:
    _cmd = ["samtools", "index", reads.local_path]
    subprocess.run(_cmd)


@large_gpu_task
def execute_run_deepvariant(
    model_type: ModelType,
    ref: LatchFile,
    reads: LatchFile,
    output_vcf: str,
    output_dir: Optional[LatchDir],
    num_shards: Optional[int],
    regions: Optional[str],
    sample_name: Optional[str],
    intermediate_results_dir: Optional[LatchDir],
    logging_dir: Optional[LatchDir],
    customized_model: Optional[LatchFile],
    make_examples_extra_args: Optional[str],
    call_variants_extra_args: Optional[str],
    postprocess_variants_extra_args: Optional[str],
    output_gvcf: Optional[str],
    vcf_stats_report: Optional[bool],
) -> List[LatchDir]:
    index_fasta(ref=ref)
    sort_reads(reads=reads)
    index_reads(reads=reads)

    if output_dir is None:
        local_out_dir = "/root/deepvariant-output/"
        local_path = Path(local_out_dir)
        if not local_path.exists():
            local_path.mkdir(exist_ok=True)
        remote_out_dir = "latch:///deepvariant-output/"

    else:
        local_out_dir = str(Path(output_dir).resolve())
        if local_out_dir[-1] != "/":
            local_out_dir += "/"
        remote_out_dir = output_dir.remote_path

    # Build required arguments
    _run_dv_cmd = [
        "run_deepvariant",
        "--model_type",
        model_type.value,
        "--ref",
        ref.local_path,
        "--reads",
        reads.local_path,
        "--output_vcf",
        local_out_dir + str(output_vcf) + ".vcf",
    ]

    if num_shards is not None:
        _run_dv_cmd.extend(["--num_shards", str(num_shards)])
    if regions is not None:
        _run_dv_cmd.extend(["--regions", regions])
    if sample_name is not None:
        _run_dv_cmd.extend(["--sample_name", sample_name])
    if intermediate_results_dir is not None:
        _run_dv_cmd.extend(["--intermediate_results_dir", intermediate_results_dir.local_path])
    if logging_dir is not None:
        _run_dv_cmd.extend(["--logging_dir", logging_dir.local_path])
    if customized_model is not None:
        _run_dv_cmd.extend(["--customized_model", customized_model.local_path])
    if make_examples_extra_args is not None:
        _run_dv_cmd.extend(["--make_examples_extra_args", make_examples_extra_args])
    if call_variants_extra_args is not None:
        _run_dv_cmd.extend(["--call_variants_extra_args", call_variants_extra_args])
    if postprocess_variants_extra_args is not None:
        _run_dv_cmd.extend(["--postprocess_variants_extra_args", postprocess_variants_extra_args])
    if output_gvcf is not None:
        _run_dv_cmd.extend(["--output_gvcf", local_out_dir + str(output_gvcf) + ".gvcf"])
    if vcf_stats_report is not None:
        if vcf_stats_report is True:
            _run_dv_cmd.extend(["--vcf_stats_report=true"])

    subprocess.run(_run_dv_cmd)
    outputs = [LatchDir(local_out_dir, remote_out_dir)]
    if logging_dir is not None:
        outputs.append(LatchDir(logging_dir.local_path, logging_dir.remote_path))
    if intermediate_results_dir is not None:
        outputs.append(LatchDir(intermediate_results_dir.local_path, intermediate_results_dir.remote_path))

    return outputs


@workflow
def deepvariant_workflow(
    model_type: ModelType,
    ref: LatchFile,
    reads: LatchFile,
    output_vcf: str,
    output_dir: Optional[LatchDir],
    num_shards: Optional[int],
    regions: Optional[str],
    sample_name: Optional[str],
    intermediate_results_dir: Optional[LatchDir],
    logging_dir: Optional[LatchDir],
    customized_model: Optional[LatchFile],
    make_examples_extra_args: Optional[str],
    call_variants_extra_args: Optional[str],
    postprocess_variants_extra_args: Optional[str],
    output_gvcf: Optional[str],
    vcf_stats_report: Optional[bool],
) -> List[LatchDir]:
    """
    DeepVariant
    ----
    ### Overview
    Deepvariant (https://github.com/google/deepvariant) is a variant caller powered by deep learning. It accepts multiple sequence alignments
    in BAM or CRAM format, and generates images that represent the data known as pileup image tensors. These image tensors are passed to a
    convolutional neural network for variant calling, with results reported in a VCF file.

    ### Inputs
    The minimum inputs required by this workflow are:
    1) A reference .fasta file with associated .fai index and
    2) Multiple sequence alignment in BAM or CRAM format with associated .bai or .crai index files.

    A model type (one of "WGS", "WES", "PACBIO", or "PACBIO_ILLUMINA_HYBRID") and output vcf filename must also be supplied.

    ### Outputs
    This workflow outputs VCF- and/or gVCF-format output files with called variants. Intermediate output (image tensors) may be optionally saved.

    ### Deepvariant License
    Copyright 2020 Google LLC.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



    __metadata__:
        display_name: DeepVariant
        author:
            name: Shea Lambert
            email:
            github: https://github.com/SheaML
        repository: https://github.com/SheaML/latch-deepvariant
        license:
            id: BSD3


    Args:
        model_type:
            The type of input data.

            __metadata__:
                display_name: Model Type

        ref:
            A reference .fasta file to which sequences have been aligned

            __metadata__:
                display_name: Reference genome
                batch_table_column: True

        reads:
            Multiple sequence alignment in .bam or .cram format

            __metadata__:
                display_name: MSA (.bam, .cram)
                batch_table_column: True

        output_vcf:
            The name for the output .vcf file

            __metadata__:
                display_name: Output VCF Filename
                batch_table_column: True

        output_dir:
            Custom Output Directory, results otherwise written to
            "/deepvariant-output/"

            __metadata__:
                display_name: Custom Output Directory
                batch_table_column: True

        num_shards:
            The number of cores to use during the make_examples step

            __metadata__:
                display_name: Shards

        regions:
            A comma-separated list of regions (e.g,. "chr20:10,000,000-10,010,000",...)
            to call variants within

            __metadata__:
                display_name: Regions
                batch_table_column: True

        sample_name:
            Sample name to use instead of the sample name from the input reads BAM
            (SM tag in the header). This flag is used for both make_examples and
            postprocess_variants.

            __metadata__:
                display_name: Sample Name
                batch_table_column: True

        intermediate_results_dir:
            Directory for storing intermediate outputs

            __metadata__:
                display_name: Intermediate Results Directory

        logging_dir:
            Directory for storing log files

            __metadata__:
                display_name: Logging Directory


        customized_model:
            Path to a custom-trained model to be used instead of one of the four defaults

            __metadata__:
                display_name: Customized Model Path

        make_examples_extra_args:
            A comma-separated list of flag_name=flag_value. "flag_name" has to be
            valid flags for make_examples.py. If the flag_value is boolean, it has to
            be flag_name=true or flag_name=false.

            __metadata__:
                display_name: Extra arguments for make_examples

        call_variants_extra_args:
            A comma-separated list of flag_name=flag_value. "flag_name" has to be
            valid flags for call_variants.py. If the flag_value is boolean, it has to
            be flag_name=true or flag_name=false.

            __metadata__:
                display_name: Extra arguments for call_variants

        postprocess_variants_extra_args:
            A comma-separated list of flag_name=flag_value. "flag_name" has to be
            valid flags for postprocess_variants.py. If the flag_value is boolean, it has to
            be flag_name=true or flag_name=false.

            __metadata__:
                display_name: Extra arguments for postprocess_variants

        output_gvcf:
            The name for the output .gvcf file

            __metadata__:
                display_name: Output gVCF Filename
                batch_table_column: True

        vcf_stats_report:
            Whether to include an .html report on the vcf output

            __metadata__:
                display_name: VCF Stats Report
    """

    return execute_run_deepvariant(
        model_type=model_type,
        ref=ref,
        reads=reads,
        output_vcf=output_vcf,
        output_dir=output_dir,
        num_shards=num_shards,
        regions=regions,
        sample_name=sample_name,
        intermediate_results_dir=intermediate_results_dir,
        logging_dir=logging_dir,
        customized_model=customized_model,
        make_examples_extra_args=make_examples_extra_args,
        call_variants_extra_args=call_variants_extra_args,
        postprocess_variants_extra_args=postprocess_variants_extra_args,
        output_gvcf=output_gvcf,
        vcf_stats_report=vcf_stats_report,
    )


## Test Data
LaunchPlan(
    deepvariant_workflow,
    "Test Data - WGS Quickstart",
    {
        "model_type": ModelType.WGS,
        "ref": LatchFile("s3://latch-public/test-data/14566/ucsc.hg19.chr20.unittest.fasta"),
        "reads": LatchFile("s3://latch-public/test-data/14566/NA12878_S1.chr20.10_10p1mb.bam"),
        "output_vcf": "quickstart_output",
    },
)
