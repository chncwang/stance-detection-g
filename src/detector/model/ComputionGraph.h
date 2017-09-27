#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "ConditionalLSTM.h"
#include "Utf.h"


// Each model consists of two parts, building neural graph and defining output losses.
class GraphBuilder {
public:
    vector<LookupNode> _inputNodes;
    ConditionalLSTMBuilder _left2right;
    ConditionalLSTMBuilder _right2left;
    LSTM1Builder _left2right_tweet;
    LSTM1Builder _right2left_tweet;
    std::vector<ConcatNode> _common_concat_nodes;
    MaxPoolNode _common_pool;

    std::vector<ConcatNode> _tweet_concat_nodes;
    std::vector<ConcatNode> _target_concat_nodes;
    MaxPoolNode _target_pooling;
    UniNode _neural_output;
    UniNode _target_output;
    AttentionBuilder _attention_builder;
    GrlNode _grl_node;
    GrlNode _target_ratio_node;
    std::vector<GrlNode> _input_bp_control_nodes;

    ConcatNode _common_domain_concat_node;
    
    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 1024;

public:
    //allocate enough nodes
    void createNodes(int length_upper_bound) {
        _inputNodes.resize(length_upper_bound);
        _left2right.resize(length_upper_bound);
        _right2left.resize(length_upper_bound);
        _left2right_tweet.resize(length_upper_bound);
        _right2left_tweet.resize(length_upper_bound);
        _tweet_concat_nodes.resize(length_upper_bound);
        _target_concat_nodes.resize(length_upper_bound);
        _common_concat_nodes.resize(length_upper_bound);
        _target_pooling.setParam(length_upper_bound);
        _attention_builder.resize(length_upper_bound);
        _common_pool.setParam(length_upper_bound);
        _input_bp_control_nodes.resize(length_upper_bound);
    }

public:
    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _inputNodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }

        for (auto &n : _input_bp_control_nodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.ratio = 0;
        }

        _left2right.init(opts.dropProb, &model.target_left_to_right_lstm_params, true);
        _right2left.init(opts.dropProb, &model.target_right_to_left_lstm_params, false);

        _left2right_tweet.init(&model.noncond_tweet_left_to_right_lstm_params, opts.dropProb, true);
        _right2left_tweet.init(&model.noncond_tweet_right_to_left_lstm_params, opts.dropProb, false);

        for (auto &n : _common_concat_nodes) {
            n.init(opts.hiddenSize * 2, -1);
        }

        _grl_node.ratio = opts.grlRatio;
        _grl_node.init(opts.hiddenSize * 2, -1);
        _target_output.setParam(&model.target_linear);
        for (auto &n : _target_concat_nodes) {
            n.init(opts.hiddenSize * 2, -1);
        }

        for (auto &n : _tweet_concat_nodes) {
            n.init(opts.hiddenSize * 2, -1);
        }
        _target_pooling.init(opts.hiddenSize * 2, -1);
        _target_output.init(DOMAIN_SIZE, -1);

        _target_ratio_node.ratio = opts.targetRatio;
        _target_ratio_node.init(DOMAIN_SIZE, -1);

        _neural_output.setParam(&model.olayer_linear);
        _neural_output.init(opts.labelSize, -1);
        _attention_builder.init(&model.attention_params);
        _common_pool.init(opts.hiddenSize * 2, -1);
        _common_domain_concat_node.init(opts.hiddenSize * 4, -1);
        _modelParams = &model;
    }

public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void forward(const Feature &feature, bool bTrain = false) {
        _graph->train = bTrain;

        const std::vector<std::string> &target_words = getStanceTargetWords(feature.m_target);

        for (int i = 0; i < target_words.size(); ++i) {
            _inputNodes.at(i).forward(_graph, target_words.at(i));
        }

        for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
            _inputNodes.at(i + target_words.size()).forward(_graph, feature.m_tweet_words.at(i));
        }

        for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
            _input_bp_control_nodes.at(i).forward(_graph, &_inputNodes.at(i + target_words.size()));
        }

        vector<PNode> inputNodes;
        int totalSize = feature.m_tweet_words.size() + target_words.size();
        for (int i = 0; i < totalSize; ++i) {
            inputNodes.push_back(&_inputNodes.at(i));
        }

        _left2right.setParam(&_modelParams->target_left_to_right_lstm_params, &_modelParams->tweet_left_to_right_lstm_params, target_words.size());
        _right2left.setParam(&_modelParams->target_right_to_left_lstm_params, &_modelParams->tweet_right_to_left_lstm_params, target_words.size());

        _left2right.forward(_graph, inputNodes, target_words.size());
        _right2left.forward(_graph, inputNodes, target_words.size());

        for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
            _tweet_concat_nodes.at(i).forward(_graph, &_left2right._hiddens.at(target_words.size() + i),
                &_right2left._hiddens.at(target_words.size() + i));
        }

        for (int i = 0; i < target_words.size(); ++i) {
            _target_concat_nodes.at(i).forward(_graph, &_left2right._hiddens.at(i), &_right2left._hiddens.at(i));
        }

        _target_pooling.forward(_graph, toPointers<ConcatNode, Node>(_target_concat_nodes, target_words.size()));

        std::vector<PNode> input_control_nodes = toPointers<GrlNode, Node>(_input_bp_control_nodes, feature.m_tweet_words.size());

        _left2right_tweet.forward(_graph, input_control_nodes);
        _right2left_tweet.forward(_graph, input_control_nodes);

        for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
            _common_concat_nodes.at(i).forward(_graph, &_left2right_tweet._hiddens.at(i), &_right2left_tweet._hiddens.at(i));
        }
        _common_pool.forward(_graph, toPointers<ConcatNode, Node>(_common_concat_nodes, feature.m_tweet_words.size()));
        _grl_node.forward(_graph, &_common_pool);
        _common_domain_concat_node.forward(_graph, &_common_pool, &_attention_builder._hidden);

        _attention_builder.forward(_graph, toPointers<ConcatNode, Node>(_tweet_concat_nodes, feature.m_tweet_words.size()), &_target_pooling);
        _target_output.forward(_graph, &_grl_node);
        _target_ratio_node.forward(_graph, &_target_output);
        _neural_output.forward(_graph, &_common_domain_concat_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
