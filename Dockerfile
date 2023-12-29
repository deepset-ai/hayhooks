FROM deepset/haystack:base-main

RUN pip install hayhooks

EXPOSE 1416

CMD ["hayhooks", "run", "0.0.0.0"]
