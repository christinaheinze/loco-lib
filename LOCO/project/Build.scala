import sbt._

object LOCOBuild extends Build {
  lazy val LOCOBuild = Project("LOCOBuild", file(".")) aggregate(preprocessingBuild) dependsOn(preprocessingBuild)
  lazy val preprocessingBuild = RootProject(file("../preprocessingUtils"))
}