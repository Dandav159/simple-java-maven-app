FROM maven:3.9-eclipse-temurin-17 AS builder
ARG APP_VERSION=1.0.0
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn clean package -Dversion=${APP_VERSION}

FROM eclipse-temurin:17.0.12_7-jre-jammy
ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}
WORKDIR /app
COPY --from=builder /app/target/my-app-${APP_VERSION}.jar ./app.jar
CMD ["java", "-jar", "my-app-${APP_VERSION}.jar"]