# Import task functions from our workflow
from wf import execute_run_deepvariant, ModelType
from latch.types import LatchFile

# Call task function
execute_run_deepvariant(
    model_type=ModelType.WGS,
    ref=LatchFile('latch:///quickstart-testdata/ucsc.hg19.chr20.unittest.fasta'),
    reads=LatchFile('latch:///quickstart-testdata/NA12878_S1.chr20.10_10p1mb.bam'),
    output_vcf="quickstart_output",
    output_dir=None,
    num_shards=None,
    regions=None,
    sample_name=None,
    intermediate_results_dir=None,
    logging_dir=None,
    customized_model=None,
    make_examples_extra_args=None,
    call_variants_extra_args=None,
    postprocess_variants_extra_args=None,
    output_gvcf=None,
    vcf_stats_report=True)
