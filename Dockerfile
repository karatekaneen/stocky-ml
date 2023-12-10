FROM tensorflow/serving
ENTRYPOINT [“/usr/bin/env”]
ENV MODEL_NAME=stocky
ENV PORT=8501
COPY model/model /models/stocky
CMD tensorflow_model_server --port=8500 --rest_api_port=$PORT --model_base_path=/models/stocky --model_name=$MODEL_NAME
