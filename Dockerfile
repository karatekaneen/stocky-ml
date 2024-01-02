FROM tensorflow/serving
ENV MODEL_NAME=stocky
ENV PORT=8501
COPY model/model /models/stocky/1
RUN echo '#!/bin/bash \n\n\
	tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
	--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
	"$@"' > /usr/bin/tf_serving_entrypoint.sh \
	&& chmod +x /usr/bin/tf_serving_entrypoint.sh
