# syntax=docker/dockerfile:1
ARG OPENWEBUI_BASE_IMAGE=ghcr.io/open-webui/open-webui:v0.11.0
ARG OPENWEBUI_SOURCE_REF=v0.11.0

FROM node:22-alpine3.20 AS frontend-build
ARG OPENWEBUI_SOURCE_REF
WORKDIR /src
RUN apk add --no-cache git
RUN git clone --depth 1 --branch "${OPENWEBUI_SOURCE_REF}" \
    https://github.com/open-webui/open-webui.git .
COPY patch-suggestions.mjs /tmp/patch-suggestions.mjs
RUN node /tmp/patch-suggestions.mjs
RUN npm ci --force
ENV APP_BUILD_HASH=lv-router-ordered-suggestions
RUN npm run build

FROM ${OPENWEBUI_BASE_IMAGE}
COPY --from=frontend-build /src/build /app/build
