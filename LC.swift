//
//  LC.swift
//  TFPlayground
//
//  Created by aaron on 2018/9/4.
//  Copyright © 2018 aaron. All rights reserved.
//

import Foundation
import TensorFlow
import Python
typealias FloatTensor = Tensor<Float>

enum Label : Int{
    case Green = 0
    case Red
}

struct Position{
    let x0 : Float = 1
    let x1 : Float
    let x2 : Float
}

struct ClassifierParameters : ParameterAggregate {
    var w = Tensor<Float>(randomNormal: [3,1])
}

struct Model
{
    var parameters : ClassifierParameters = ClassifierParameters()
}


func loadTrainingSet() -> (trainingVec : FloatTensor , labelVec : FloatTensor)
{
    let lines = try! String(contentsOf: URL(fileURLWithPath: "test.txt")).split(separator: "\r\n")
    let data = lines.map{$0.split(separator: "\t")}
    let rowCount = data.count
    let trainingScalars:[[Float]] = data.map{[1.0, Float($0[0])!, Float($0[1])!]}
    let labelScalarts:[Float] = data.map{Float($0[2])!}
    
    let trainingVec = Tensor<Float>(shape: [Int32(rowCount), 3], scalars: trainingScalars.flatMap{$0})
    let labelVec = Tensor<Float>(shape: [Int32(rowCount) , 1], scalars: labelScalarts)
    return (trainingVec, labelVec)
}


func train(trainingVec : FloatTensor, labelVec : FloatTensor, model : inout Model)
{
    let learningRate:Float = 0.0005

    for epoch in 0...3000
    {
        let y = trainingVec • model.parameters.w
        let h = sigmoid(y)
        let e = labelVec - h
        let dw = trainingVec.transposed() • e
        
        let grad = ClassifierParameters(w: dw)
        model.parameters.update(withGradients: grad) { (p, g) in
            p += g * learningRate
        }

        let p1 = -1 * labelVec * log(h)
        let p2 = (1 - labelVec)*log(1 - h)
        let traditionalLogLoss = ((p1 - p2).sum() / batchSize)

        print("epoch: \(epoch), LogLoss v2: \(traditionalLogLoss)")
    }
}

func plot(trainVec : FloatTensor, labelVec : FloatTensor, parameters : ClassifierParameters)
{
    var coord1x:[Float] = []
    var coord1y:[Float] = []
    var coord2x:[Float] = []
    var coord2y:[Float] = []
    let rowCount = trainVec.shape[0]

    for i in 0..<rowCount
    {
        if Int(labelVec[i][0].scalar!) == 1
        {
            coord1x.append(trainVec[i][1].scalar!)
            coord1y.append(trainVec[i][2].scalar!)
        }
        else
        {
            coord2x.append(trainVec[i][1].scalar!)
            coord2y.append(trainVec[i][2].scalar!)
        }
        print(i)
    }

    var xpts = Array<Float>(repeating: 0, count: 60)
    for i in 0..<xpts.count
    {
        xpts[i] = -3 + Float(i)*0.1
    }
    print(parameters.w)
    let wVec = parameters.w.scalars
    let w1 = wVec[0]
    let w2 = wVec[1]
    let w3 = wVec[2]

    let ypts:[Float] = xpts.map{ (x) -> Float in
        let a = -w1
        let b = w2 * x
        let c = w3
        let d = (a - b) / c
        return d
    }
    let matplot = Python.import("matplotlib.pyplot")

    let fig = matplot.figure()
    let ax = fig.add_subplot(111)
    ax.scatter(coord1x, coord1y, 50, "red", "s")
    ax.scatter(coord2x,coord2y,50, "green")
    ax.plot(xpts, ypts)
    matplot.show()
}

typealias Probability = Float


func launch()
{
    typealias Record = (Position , Label)
    let (trainingVec, labelVec) = loadTrainingSet()
    var m = Model()
    train(trainingVec: trainingVec, labelVec: labelVec, model: &m)
    print(m)
    plot(trainVec: trainingVec, labelVec: labelVec, parameters: m.parameters)
    print(m.parameters.w)
    while true {
        
    }
}

launch()

func predict(model : Model, pos : Position) -> (Label, Probability){
    return (.Green, 1)
}

