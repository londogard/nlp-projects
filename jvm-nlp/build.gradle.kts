plugins {
    java
    kotlin("jvm") version "1.3.71"
}

group = "com.londogard"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://jitpack.io")
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))

    // Smile
    implementation("com.github.haifengl:smile-core:2.3.0")
    implementation("com.github.haifengl:smile-kotlin:2.3.0")
    implementation("com.github.haifengl:smile-nlp:2.3.0")
    implementation("com.londogard:smile-nlp-kt:1.1.0")

    testImplementation("junit", "junit", "4.12")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
}