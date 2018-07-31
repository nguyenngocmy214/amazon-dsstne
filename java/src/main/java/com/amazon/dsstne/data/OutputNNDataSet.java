/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */

package com.amazon.dsstne.data;

import java.nio.ByteBuffer;

import com.amazon.dsstne.Dim;
import com.amazon.dsstne.NNDataSetEnums.DataType;

import lombok.Getter;

/**
 * Data set to hold outputs of the predictions from a network.
 * Output data is comprised of two datasets: indexes and scores.
 * The indexes hold the indexes of the top-k results.
 * The scores hold the output values of the top-k results.
 */
public class OutputNNDataSet {

  @Getter
  private final Dim dim;
  private final DenseNNDataSet indexes;
  private final DenseNNDataSet scores;

  public OutputNNDataSet(final Dim dim) {
    this.dim = dim;
    this.indexes = new DenseNNDataSet(dim, DataType.LLInt);
    this.scores = new DenseNNDataSet(dim, DataType.Float);
  }

  public ByteBuffer getIndexesData() {
    return indexes.getData();
  }

  public ByteBuffer getScoresData() {
    return scores.getData();
  }
}

