name := "LOCO"

version := "0.1.1"

scalaVersion := "2.10.4"

// additional libraries
{
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.4.1" % "provided",
  "org.apache.spark"  %% "spark-mllib" % "1.4.1",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "cc.factorie" % "factorie_2.10" % "1.1.1",
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "com.nativelibs4java" % "jnaerator" % "0.11",
  "net.java.dev.jna" % "jna" % "3.4.0"
)
}

resolvers ++= Seq(
  "IESL Release" at "http://dev-iesl.cs.umass.edu/nexus/content/groups/public",
   "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

// Configure jar named used with the assembly plug-in
assemblyJarName in assembly := "LOCO-assembly-0.1.1.jar"

// assembly merge strategy
assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*)           => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".html"   => MergeStrategy.first
  case "application.conf"                              => MergeStrategy.concat
  case "reference.conf"                                => MergeStrategy.concat
  case "log4j.properties"                              => MergeStrategy.discard
  case m if m.toLowerCase.endsWith("manifest.mf")      => MergeStrategy.discard
  case m if m.toLowerCase.matches("meta-inf.*\\.sf$")  => MergeStrategy.discard
  case _ => MergeStrategy.first
}
