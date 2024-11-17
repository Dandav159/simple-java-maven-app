FROM maven:latest AS builder

ARG APP_VERSION=1.0.0

WORKDIR /app
COPY . .

RUN mvn clean package

FROM eclipse-temurin:17.0.12_7-jre-jammy

ENV APP_VERSION=1.0.0

WORKDIR /app

COPY --from=builder /app/target .

CMD ["ls"]