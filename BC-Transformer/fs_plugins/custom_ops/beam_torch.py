#Author:SongSheng Wang
#Date: 7/7/2023
#Implement the beam search without lenpen.
#Notice that there may be a bug in this code, since the decoded beam BLEU is not good.
#Efficiency also need to be improved. 7s / sentence, which is tooooo slow.
#usage:
'''start = time.time()
                res,score=self._dag_search_torch(dagscores, nextstep_idx, logits_idx,
                    output_length,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    decode_direction = self.args.decode_direction,
                    top_logits=top_logits
                )
                end = time.time()
                print('dag search time:',end-start)'''
#func in model:
    def __get_best_sentence(self,beam,beam_step_idx,beam_logits,prelen):
            bsz = len(beam)
            best_sentence = []
            best_logits = []
            best_step = []
            for batch in range(bsz):
                max_idx = np.argmax(np.array(beam_logits[batch]))
                best_sentence.append(beam[batch][max_idx])
                best_logits.append(beam_logits[batch][max_idx]*prelen)
                best_step.append(beam_step_idx[batch][max_idx])
            for i in range(bsz):
                if best_sentence[i][-1] != self.tgt_dict.eos():
                    best_sentence[i].append(self.tgt_dict.eos())
                pad_num = prelen - len(best_sentence[i])
                for num in range(pad_num):
                    best_sentence[i].append(self.tgt_dict.pad())
            return best_sentence,best_logits
    def __end_loop(self,beam_status_array):
        batch_status = []
        for beam in beam_status_array:
            if True not in beam:
                batch_status.append(False)
            else:
                batch_status.append(True)
        if True not in batch_status:
            return True
        else:
            return False
    def __shrink_beam(self,beam,beam_logits,beam_idx,beam_size,beam_status):
        index = np.argsort(beam_logits)
        return_beam = []
        return_idx = []
        return_logits = []
        return_status = []
        for i in range(beam_size):
            return_beam.append(beam[index[i]])
            return_idx.append(beam_idx[index[i]])
            return_logits.append(beam_logits[index[i]])
            return_status.append(beam_status[index[i]])
        return return_beam,return_logits,return_idx,return_status
    def _dag_search_torch(self,dagscores,nextstep_idx,logits_idx,output_length,
                          alpha,gamma,beamsize,max_beam_per_length,decode_top_p,decode_direction='forward',top_logits=None):
        #实现一个beam_search算法，可以支持以下功能：
        #参数逐步添加
        #Forward Search
        batch_size,prelen,topk = dagscores.shape
        if decode_direction == 'forward':
            #初始化 beam: token id list  beam_logits: sentence logits list beam_status: sentence search status list (true for search, false for seach finished) 
            #这里每个beam有五个句子，每个句子有五个位置零最可能的token。
            beam = [] 
            beam_step_idx = [] #代表这个sample的路径，最后一个id是需要去的下一个位置，搜索完了最后一个id就是-1.
            beam_logits = [] 
            beam_status = []
            for batch in range(batch_size):
                temp_beam = []
                temp_beam_logits = []
                temp_beam_step_idx = []
                temp_beam_status = []
                for top in range(self.args.decode_top_cand_n):
                    #beam原来就是个空列表，这里应该append第一个token的id。
                    temp_beam.append([self.tgt_dict.bos(),logits_idx[batch][0][top]])
                    temp_beam_logits.append(top_logits[batch][0][top])
                    temp_beam_step_idx.append([0])
                    temp_beam_status.append(True)
                beam.append(temp_beam)
                beam_step_idx.append(temp_beam_step_idx)
                beam_logits.append(temp_beam_logits)
                beam_status.append(temp_beam_status)
            #现在每个batch的初始化完成了，是第一个token的最可能的五个id和logits。
            #后面每一个step需要先加入五个position，再加入五个token。
            #----------------------------Code Needs To Implement-------------------------------------------
            end_tokens = [self.tgt_dict.eos(),self.tgt_dict.bos(),self.tgt_dict.pad()]
            #print('prelen',prelen)
            for step in range(prelen-2):
                #print('step,',step)
                #expand the beam candidate
                for batch in range(batch_size):
                    #print('batch',batch)
                    #对于每个candidate 应该先存下来，然后expand
                    if True not in beam_status[batch]:
                        continue
                    sentence_candidate = beam[batch].copy()
                    sentence_step_candidate = beam_step_idx[batch].copy()
                    new_beam = []
                    new_beam_logits = []
                    new_beam_step_idx = []
                    new_beam_status = []
                    for sentence_idx in range (len(sentence_candidate)):
                        #if all the search failed, the sentence stop searching.
                        sentence_search_flag = False
                        #print('sentence_idx',sentence_idx)
                        sentence = sentence_candidate[sentence_idx]
                        sentence_logits = beam_logits[batch][sentence_idx]
                        
                        if beam_status[batch][sentence_idx] == False:
                            new_beam.append(sentence)
                            new_beam_logits.append(sentence_logits)
                            new_beam_step_idx.append(sentence_step_candidate[sentence_idx])
                            new_beam_status.append(False)
                            continue
                        #先拓展5个位置
                        for pos_id in range(len(nextstep_idx[batch][beam_step_idx[batch][sentence_idx][-1]])):
                            #优化：针对重复pos，只搜索第一个
                            if pos_id > 1:
                                if nextstep_idx[batch][beam_step_idx[batch][sentence_idx][-1]][pos_id] == nextstep_idx[batch][beam_step_idx[batch][sentence_idx][-1]][pos_id-1]:
                                    continue
                            pos = nextstep_idx[batch][beam_step_idx[batch][sentence_idx][-1]][pos_id]
                            if pos >= prelen:
                                print('ERROR Wrong POS:')
                                print(pos)
                                exit()
                        #再拓展5*5个tokens
                            for top in range(self.args.decode_top_cand_n):
                                if dagscores[batch][pos][top] == float('-inf'):
                                    continue
                                #优化：针对重复tokens的情况，只搜索第一个。
                                if top > 1:
                                    if (logits_idx[batch][pos][top] == logits_idx[batch][pos][top-1]) & (dagscores[batch][pos][top]==dagscores[batch][pos][top-1]):
                                        continue
                                else:
                                    sentence_search_flag = True
                                temp_sentence_step_candidate = sentence_step_candidate[sentence_idx].copy()
                                temp_sentence_step_candidate.append(pos)
                                new_beam_step_idx.append(temp_sentence_step_candidate)
                                tmp_sentence = sentence.copy()
                                tmp_sentence.append(logits_idx[batch][pos][top])
                                #print(logits_idx[batch][pos])
                                #print(tmp_sentence)
                                if tmp_sentence[-1] in end_tokens:
                                    new_beam_status.append(False)
                                else:
                                    new_beam_status.append(True)
                                new_beam.append(tmp_sentence)
                        #最后算对应的logits
                                new_beam_logits.append(beam_logits[batch][sentence_idx] + dagscores[batch][pos][top])
                                #print(dagscores[batch][pos][top])
                                #print(beam_logits[batch][sentence_idx])
                        if sentence_search_flag == False:
                            new_beam.append(sentence)
                            new_beam_logits.append(sentence_logits)
                            new_beam_step_idx.append(sentence_step_candidate[sentence_idx])
                            new_beam_status.append(False)
                    '''print('new beam')
                    for item in new_beam:
                        print(item)
                    print('now beam logits')
                    for item in new_beam_logits:
                        print(item)'''
                    #之后看是不是超过了beam_size，如果超过的话还要减小。
                    if len(new_beam_logits) > beamsize:
                        new_beam,new_beam_logits,new_beam_step_idx,new_beam_status = self.__shrink_beam(new_beam,new_beam_logits,new_beam_step_idx,beamsize,new_beam_status)
                    '''print('shrinked new beam')
                    for item in new_beam:
                        print(item)
                    print('shrinked now beam logits')
                    for item in new_beam_logits:
                        print(item)'''
                    end_tokens = [self.tgt_dict.eos(),self.tgt_dict.bos(),self.tgt_dict.pad()]
                    #print('beam status')
                    #print(new_beam_status)
                    beam[batch] = new_beam
                    beam_step_idx[batch] = new_beam_step_idx
                    beam_logits[batch] = new_beam_logits
                    beam_status[batch] = new_beam_status
                if  self.__end_loop(beam_status):
                    break
            res,score = self.__get_best_sentence(beam,beam_step_idx,beam_logits,prelen)
            return res, score
        #--------------------------------------------------------------------------------------------------------------------------------
        elif decode_direction == 'backward':
            print('ERROR! Not Implemented')
            exit()
            return
        elif decode_direction == 'joint':
            if top_logits==None:
                print('ERROR! No token logits for joint beam decoding!')
                exit()
            else:
                print("ERROR! Joint Search Not Implemented")
                exit()
                return
        else:
            print('ERROR! Wrong Decoding Direction! Should be forward/backward/joint')
            print(decode_direction)
            exit()