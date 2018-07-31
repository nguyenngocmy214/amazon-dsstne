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
#include <dlfcn.h>
#include <sstream>

#include "amazon/dsstne/engine/GpuTypes.h"
#include "amazon/dsstne/engine/NNTypes.h"
#include "amazon/dsstne/engine/NNLayer.h"

#include "jni_util.h"
#include "com_amazon_dsstne_Dsstne.h"

using namespace dsstne;

namespace
{
const unsigned long SEED = 12134ull;

void *LIB_MPI = NULL;
const char* LIB_MPI_SO = "libmpi.so";

const int ARGC = 1;
char *ARGV = "jni-faux-process";

jni::References REFS;

jmethodID java_ArrayList_;
jmethodID java_ArrayList_add;

jmethodID dsstne_NNLayer_;

jmethodID dsstne_NNDataSet_getAttribute;
jmethodID dsstne_NNDataSet_getDataTypeOrdinal;
jmethodID dsstne_NNDataSet_getSparseStart;
jmethodID dsstne_NNDataSet_getSparseEnd;
jmethodID dsstne_NNDataSet_getSparseIndex;
jmethodID dsstne_NNDataSet_getData;

jmethodID dsstne_OutputNNDataSet_getIndexesData;
jmethodID dsstne_OutputNNDataSet_getScoresData;

GpuContext* checkPtr(JNIEnv *env, jlong ptr)
{
    GpuContext *gpuContext = (GpuContext*) ptr;
    if (gpuContext == NULL)
    {

        jni::throwJavaException(env, jni::RuntimeException,
                                "GpuContext pointer is null, call init() prior to any other functions");
    }
    return gpuContext;
}
}

jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    /*
     * JVM loads dynamic libs into a local namespace,
     * MPI requires to be loaded into a global namespace
     * so we manually load it into a global namespace here
     */
    LIB_MPI = dlopen(LIB_MPI_SO, RTLD_NOW | RTLD_GLOBAL);

    if (LIB_MPI == NULL)
    {
        std::cerr << "Failed to load libmpi.so" << std::endl;
        exit(1);
    }

    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
    {
        return JNI_ERR;
    } else
    {
//    jni::findClassGlobalRef(env, REFS, jni::NNLayer);
//    jni::findClassGlobalRef(env, REFS, jni::ArrayList);

        java_ArrayList_ = jni::findConstructorId(env, REFS, jni::ArrayList, jni::NO_ARGS_CONSTRUCTOR);
        java_ArrayList_add = jni::findMethodId(env, REFS, jni::ArrayList, "add", "(Ljava/lang/Object;)Z");

        dsstne_NNLayer_ = jni::findConstructorId(env, REFS, jni::NNLayer,
                                                 "(Ljava/lang/String;Ljava/lang/String;IIIIII)V");

        dsstne_NNDataSet_getAttribute = jni::findMethodId(env, REFS, jni::NNDataSet, "getAttribute", "()I");
        dsstne_NNDataSet_getDataTypeOrdinal = jni::findMethodId(env, REFS, jni::NNDataSet, "getDataTypeOrdinal", "()I");
        dsstne_NNDataSet_getSparseStart = jni::findMethodId(env, REFS, jni::NNDataSet, "getSparseStart", "()[J");
        dsstne_NNDataSet_getSparseEnd = jni::findMethodId(env, REFS, jni::NNDataSet, "getSparseEnd", "()[J");
        dsstne_NNDataSet_getSparseIndex = jni::findMethodId(env, REFS, jni::NNDataSet, "getSparseIndex", "()[J");
        dsstne_NNDataSet_getData = jni::findMethodId(env, REFS, jni::NNDataSet, "getData", "()Ljava/nio/ByteBuffer;");

        dsstne_OutputNNDataSet_getIndexesData = jni::findMethodId(env, REFS, jni::OutputNNDataSet, "getIndexesData",
                                                                  "()Ljava/nio/ByteBuffer;");
        dsstne_OutputNNDataSet_getScoresData = jni::findMethodId(env, REFS, jni::OutputNNDataSet, "getScoresData",
                                                                 "()Ljava/nio/ByteBuffer;");

        return JNI_VERSION_1_6;
    }
}

void JNI_OnUnload(JavaVM *vm, void *reserved)
{
    using namespace dsstne;

    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK)
    {
        return;
    } else
    {
        jni::deleteReferences(env, REFS);
    }
}

JNIEXPORT jlong JNICALL Java_com_amazon_dsstne_Dsstne_load(JNIEnv *env, jclass clazz, jstring jNetworkFileName,
                                                           jint batchSize)
{
    const char *networkFileName = env->GetStringUTFChars(jNetworkFileName, 0);

    getGpu().Startup(ARGC, &ARGV);
    getGpu().SetRandomSeed(SEED);
    NNNetwork *network = LoadNeuralNetworkNetCDF(networkFileName, batchSize);
    getGpu().SetNeuralNetwork(network);

    env->ReleaseStringUTFChars(jNetworkFileName, networkFileName);
    return (jlong) &getGpu();
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_loadDatasets(JNIEnv *env, jclass clazz, jlong ptr,
                                                                  jobjectArray datasetNames, jintArray attributes,
                                                                  jintArray dataTypes, jintArray dimXs, jintArray dimYs,
                                                                  jintArray dimZs, jintArray examples)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    NNNetwork *network = gpuContext->_pNetwork;

    // parameters are aligned by index; all arrays have same length
    jsize len = env->GetArrayLength(datasetNames);
    jint *jAttributes = env->GetIntArrayElements(attributes, NULL);
    jint *jDataTypes = env->GetIntArrayElements(dataTypes, NULL);
    jint *jDimXs = env->GetIntArrayElements(dimXs, NULL);
    jint *jDimYs = env->GetIntArrayElements(dimYs, NULL);
    jint *jDimZs = env->GetIntArrayElements(dimZs, NULL);
    jint *jExamples = env->GetIntArrayElements(examples, NULL);

    for(jsize i = 0; i < len; ++i) {
        jstring jDatasetName = (jstring) env->GetObjectArrayElement(datasetNames, i);
        const char *datasetName = env->GetStringUTFChars();
        int attribute = jAttributes[i];
        int dataType = jDataTypes[i];
        int dimX = jDimXs[i];
        int dimY = jDimYs[i];
        int dimZ = jDimZs[i];
        int example = jExamples[i];

        env->ReleaseStringUTFChars(jDatasetName, datasetName);

    }

    env->ReleaseIntArrayElements(attributes, jAttributes, JNI_ABORT);
    env->ReleaseIntArrayElements(dataTypes, jDataTypes, JNI_ABORT);
    env->ReleaseIntArrayElements(dimXs, jDimXs, JNI_ABORT);
    env->ReleaseIntArrayElements(dimYs, jDimYs, JNI_ABORT);
    env->ReleaseIntArrayElements(dimZs, jDimZs, JNI_ABORT);
    env->ReleaseIntArrayElements(examples, jExamples, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_shutdown(JNIEnv *env, jclass clazz, jlong ptr)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    gpuContext->Shutdown();
}

JNIEXPORT jobject JNICALL Java_com_amazon_dsstne_Dsstne_get_1layers(JNIEnv *env, jclass clazz, jlong ptr,
                                                                    jint kindOrdinal)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    NNNetwork *network = gpuContext->_pNetwork;
    NNLayer::Kind kind = static_cast<NNLayer::Kind>(kindOrdinal);

    std::vector<const NNLayer*> layers;
    std::vector<const NNLayer*>::iterator it = network->GetLayers(kind, layers);
    if (it == layers.end())
    {
        jni::throwJavaException(env, jni::RuntimeException, "No layers of type %s found in network %s",
                                NNLayer::_sKindMap[kind], network->GetName());
    }

    jobject jLayers = jni::newObject(env, REFS, jni::ArrayList, java_ArrayList_);

    for (; it != layers.end(); ++it)
    {
        const NNLayer* layer = *it;
        const std::string &name = layer->GetName();
        const std::string &datasetName = layer->GetDataSetName();
        jstring jName = env->NewStringUTF(name.c_str());
        jstring jDatasetName = env->NewStringUTF(datasetName.c_str());
        int kind = static_cast<int>(layer->GetKind());
        uint32_t attributes = layer->GetAttributes();

        uint32_t numDim = layer->GetNumDimensions();

        uint32_t lx, ly, lz, lw;
        std::tie(lx, ly, lz, lw) = layer->GetDimensions();

        jobject jInputLayer = jni::newObject(env, REFS, jni::NNLayer, dsstne_NNLayer_, jName, jDatasetName, kind,
                                             attributes, numDim, lx, ly, lz);

        jclass java_util_ArrayList = REFS.getClassGlobalRef(jni::ArrayList);
        env->CallBooleanMethod(jLayers, java_ArrayList_add, jInputLayer);
    }
    return jLayers;
}

JNIEXPORT void JNICALL Java_com_amazon_dsstne_Dsstne_predict(JNIEnv *env, jclass clazz, jlong ptr, jint k,
                                                             jobjectArray jInputs, jobjectArray jOutputs)
{
    GpuContext *gpuContext = checkPtr(env, ptr);
    NNNetwork *network = gpuContext->_pNetwork;
    int batchSize = network->GetBatch();

    // expect jInputs.size == network.inputLayers.size and jOutputs.size == network.outputLayers.size
    jsize inputLen = env->GetArrayLength(jInputs);
    jsize outputLen = env->GetArrayLength(jOutputs);

    for (jsize i = 0; i < inputLen; ++i)
    {
        jobject jInput = env->GetObjectArrayElement(jInputs, i);
        jobject jOutput = env->GetObjectArrayElement(jOutputs, i);

        NNDataSetEnums::DataType dataType = static_cast<NNDataSetEnums::DataType>(env->CallIntMethod(
            jInput, dsstne_NNDataSet_getDataTypeOrdinal));
        jint attribute = env->CallIntMethod(jInput, dsstne_NNDataSet_getAttribute);
        jobject jInputData = env->CallObjectMethod(jInput, dsstne_NNDataSet_getData);
        jobject jOutputData = env->CallObjectMethod(jOutput, dsstne_OutputNNDataSet_getScoresData);
        jarray jSS = (jarray) env->CallObjectMethod(jInput, dsstne_NNDataSet_getSparseStart);
        jsize len = env->GetArrayLength(jSS);
        std::cout << "SparseStart length: " << len << std::endl;
        jlong *jSparseStart = (jlong*) env->GetPrimitiveArrayCritical(jSS, NULL);

        std::cout << "Got data type: " << dataType << " attribute: " << attribute << std::endl;
        void *cInputData = env->GetDirectBufferAddress(jInputData);
        void *cOutputData = env->GetDirectBufferAddress(jOutputData);

        uint32_t lx, ly, lz, lw;
        const NNLayer* layer = network->GetLayer("Input");
        std::tie(lx, ly, lz, lw) = layer->GetDimensions();

        switch (dataType) {
            case NNDataSetEnums::DataType::Int: {
                int *ciidata = (int*) cInputData;
                float *ciodata = (float*) cOutputData;
                for (uint32_t j = 0; j < lx * batchSize; ++j)
                {
                    float d = (float) (ciidata[j]);
                    ciodata[j] = d;
                }
            }
                break;
            case NNDataSetEnums::DataType::LLInt:
                break;
            case NNDataSetEnums::DataType::Float:
                break;
            case NNDataSetEnums::DataType::Double:
                break;
            case NNDataSetEnums::DataType::Char:
                break;
            default:
                std::stringstream msg;
                msg << "Unsupported data type: " << dataType;
                jni::throwJavaException(env, jni::IllegalArgumentException, "Unsupported data type %d", dataType);
        }
    }

}

