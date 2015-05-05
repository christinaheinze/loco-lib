package LOCO.utils

import cc.factorie.la.{DenseTensor1, Tensor1}
import cc.factorie.variable.{TensorVariable, DiffList}

class Input(val observation: Array[Double])(implicit d: DiffList = null
  ) extends TensorVariable[Tensor1] {

  val p = observation.length
  val myObservation = new DenseTensor1(p)
  myObservation := observation
  set(myObservation)
}

class Example(val input : Input, val label : Double)(implicit d: DiffList = null
  ) extends TensorVariable[Tensor1]{
  set(new DenseTensor1(1))
  value(0) = label
}