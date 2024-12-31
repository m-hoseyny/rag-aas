FROM python:3.10-slim-bullseye
ENV TZ=Asia/Tehran
 
ENV LISTEN_PORT 8443
 
EXPOSE 8443
 
# RUN apt-get update && apt-get install -y git
 
COPY ./requirements.txt /app/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
WORKDIR app/
 
COPY ./ /app/
RUN chmod +x /app/entrypoint.sh
RUN ls .
 
ENTRYPOINT ["./entrypoint.sh"]