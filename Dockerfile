FROM maven:3.9-eclipse-temurin-17 AS builder
ARG APP_VERSION
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn clean package
RUN ls -la target/

FROM eclipse-temurin:17.0.12_7-jre-jammy
ARG APP_VERSION
WORKDIR /app
COPY --from=builder /app/target/my-app-*.jar ./app.jar
CMD ["java", "-jar", "app.jar"]