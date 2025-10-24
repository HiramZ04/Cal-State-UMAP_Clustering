FROM ollama/ollama:latest AS builder

COPY Modelfile /root/Modelfile

RUN /bin/sh -lc 'ollama serve & sleep 3 && \
                 ollama create cal-state-words -f /root/Modelfile && \
                 pkill ollama || true'

FROM ollama/ollama:latest
COPY --from=builder /root/.ollama /root/.ollama
EXPOSE 11434
CMD ["serve"]
