import logging

logger = logging.getLogger("lexagent.tracing")

try:
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False

def setup_tracing():
    """
    Initialize OpenTelemetry trace instrumentation for NVIDIA NIM / OpenAI endpoints,
    exporting tracing spans to local Arize Phoenix dashboard.
    """
    if not TRACING_ENABLED:
        logger.info("OpenTelemetry / Arize Phoenix tracing libraries are not installed. Skipping tracing hooks.")
        return
    
    try:
        logger.info("Initializing OpenInference tracing instrumentation...")
        # Configure TracerProvider
        provider = TracerProvider()
        
        # Default OTLP gRPC endpoint for Arize Phoenix running on localhost
        endpoint = "http://localhost:6006/v1/traces"
        
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        
        # Instrument OpenAI / NIM calls
        OpenAIInstrumentor().instrument()
        logger.info(f"Observability tracing active. Spans exporting to {endpoint}")
    except Exception as e:
        logger.error(f"Failed to initialize tracing hooks: {e}")
