import sbt._

object preprocessingBuild extends Build {
  lazy val preprocessingBuild = Project("preprocessingBuild", file("."))
}