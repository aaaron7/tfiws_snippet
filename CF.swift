//
//  Titanic.swift
//  TFPlayground
//
//  Created by aaron on 2018/8/29.
//  Copyright © 2018 aaron. All rights reserved.
//

import Foundation
import TensorFlow
import Python

enum RecommendsStrategy {
    case User2Item
    case Item2Item
}

let epsilon:Float = 1e-9

extension Array where Element : Comparable
{
    mutating func partition(_ begin : Int, _ end : Int) -> Int
    {
        let pivotIdx = begin
        let pivot = self[pivotIdx]
        
        var low = begin
        var high = end
        while (low < high)
        {
            while (low < high && self[low] <= pivot){
                low += 1
            }
            while (low < high && self[high] > pivot){
                high -= 1
            }
            
            if (low < high){
                swapAt(low, high)
            }
        }
        
        
        return low
    }
    
    mutating func topk(begin : Int , end : Int, k : Int) -> Int
    {
        let target = count - k
        let pivotIdx = partition(begin, end)
        if pivotIdx == target{
            return pivotIdx
        }else if (pivotIdx > target){
            return topk(begin: begin, end: pivotIdx - 1, k: k)
        }else{
            return topk(begin: pivotIdx + 1, end: end, k: k)
        }
    }
}



func loadData(path : String) -> (trainMatrix : Tensor<Float>, trainData : [[Float]] , testScalar : [Float])
{
    let lines = try! String(contentsOf: URL(fileURLWithPath: path)).split(separator: "\n")
    let data = lines.map{$0.split(separator: "\t")}
    let dataSet : [[Float]] = data.map{[Float($0[0])!, Float($0[1])!, Float($0[2])!]}

    let nUsers = Set(dataSet.map{$0[0]}).count
    let nMovies = Set(dataSet.map{$0[1]}).count
    
    print ("Total user is : \(nUsers), total movies are \(nMovies)")
    let rating = Array<Float>(repeating: 0, count: nMovies)
    var scalars:[[Float]] = Array<[Float]>(repeating: rating, count: nUsers)
    let ratingv2 = rating
    var testScalars:[[Float]] = Array<[Float]>(repeating: ratingv2, count: nUsers)
    
    let (trainData, testData) = sliceTrainSet(input: dataSet, ratio: 0.25)
    for item in trainData
    {
        scalars[Int(item[0]) - 1][Int(item[1]) - 1] = item[2]
    }
    
    for item in testData
    {
        testScalars[Int(item[0]) - 1][Int(item[1]) - 1] = item[2]
    }
    

    let tensor = Tensor<Float>(shape: [Int32(nUsers), Int32(nMovies)], scalars: scalars.flatMap{$0}).toAccelerator()
    return (tensor, trainData, testScalars.flatMap{$0})
}

func sliceTrainSet(input : [[Float]], ratio : Float) -> (train : [[Float]], test: [[Float]])
{
    let count = Float(input.count)
    let upTo = Int(count * (1 - ratio))
    let train = input.prefix(upTo: upTo)
    let test = input.suffix(from: upTo)
    return (Array<[Float]>(train), Array<[Float]>(test))
}

func predict(rating : Tensor<Float>, similarity : Tensor<Float>) -> Tensor<Float>
{
    let part = abs(similarity).sum(alongAxes: 1).transposed() + epsilon
    
    let result = (rating • similarity / part)
    return result
}


func predictTopk(rating : Tensor<Float>, similarity : Tensor<Float>) -> Tensor<Float>
{
        let part2 = abs(similarity).sum(alongAxes: 0) + epsilon
        let result = rating • (similarity.transposed()) / part2
        return result
}

func runCF() -> Void
{
    let (trainTensor, _ , testScalar) = loadData(path: "ml-100k/u.data")
    let bakTensor = trainTensor
    let userSimilarity = pairwiseSimilarity(x: trainTensor)
    let itemSimilarity = pairwiseSimilarity(x: trainTensor.transposed())
    let topkItemSimilarity = topkize(input: itemSimilarity, 60)
    let pred = predict(rating: trainTensor.toAccelerator(), similarity: itemSimilarity.toAccelerator())
    let predTopk = predictTopk(rating: trainTensor.toAccelerator(), similarity: topkItemSimilarity.toAccelerator())
    let i2iMSE = mse(pred: predTopk.scalars, truth: testScalar)
    let i2iMSEOriginal = mse(pred: pred.scalars, truth: testScalar)
    print("original mse is \(i2iMSEOriginal) topk mse is \(i2iMSE)")
}

func topkize(input : Tensor<Float>, _ k : Int) -> Tensor<Float>
{
    let row = input.shape[0]
    let col = input.shape[1]
    var scalar = input.scalars
    for i in 0..<row{
        let begin = Int(i * col)
        let end = Int((i+1) * col - 1)
        let slice = scalar[begin...end]
        var arr : [Float] = Array<Float>(slice)
        let kIdx = arr.topk(begin: 0, end: arr.count - 1, k: k)
        let threshold = arr[kIdx]
        for j in begin...end{
            if scalar[j] < threshold{
                scalar[j] = 0
            }
        }
    }
    
    return Tensor<Float>(shape: [row,col], scalars: scalar)
}



func mse(pred : [Float], truth : [Float]) -> Float
{
    var mse:Float = 0.0
    var count = 0
    _ = zip(Range(NSMakeRange(0, pred.count))!, truth).filter {$1 > 1e-9}
        .map { (index, truthValue) -> Float in
            mse += pow((pred[index] - truthValue), Float(2))
            count += 1
            return 0
    }
    return mse / (Float(count) + 1e-9)
}

func cosineSimilarty(t1 : Tensor<Float>, t2 : Tensor<Float>) -> Float
{
    let part1 = (t1 * t2).sum()
    let part2 = (t1 * t1).sum() * (t2 * t2).sum()
    return 1 -  part1 / sqrt(part2)
}

func pairwiseSimilarity(x : Tensor<Float>) -> Tensor<Float> {
    let sumedX = x.squared().sum(alongAxes: 1)
    return x • x.transposed() / (sqrt(sumedX • sumedX.transposed()) + epsilon)
}

runCF()
