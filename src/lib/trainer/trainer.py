import os
import csv
import json
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer(object):

    def __init__(self, **kwargs):
        self.optimizer = kwargs['optimizer']
        self.epoch = kwargs['epoch']
        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self._best_accuracy = 0.0
        self.device = kwargs['device']
        self.results = {}


    def train(self, model, train_iterator, validation_iterator):
        tester_args = {
            'save_dir' : self.save_dir,
            'file_list' : None,
            'device' : self.device,
            'label_list' : None
            }
        validator = Tester(**tester_args)
        start = time.time()
        best_epoch = 1

        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            # turn on network training mode
            model.train()

            loss_train, eval_results_train = self._train_step(model, train_iterator)
            eval_results_val, loss_val = validator.test(model, validation_iterator, phase="validation")

            self._save_log(epoch, loss_train, loss_val, eval_results_train, eval_results_val)

            if self.best_eval_result(eval_results_val):
                torch.save(model.to('cpu'), os.path.join(self.save_dir, 'best_model.npz'))
                model.to(torch.device(self.device))
                best_epoch = epoch
                print("Saved better model selected by validation.")
                with open(os.path.join(self.save_dir, 'best_result'), 'w') as f:
                    json.dump({'best {}'.format(self.eval_metrics): self._best_accuracy,
                               'best epoch': best_epoch}, f, indent=4)
        print('best {}: {}'.format(self.eval_metrics, self._best_accuracy))
        print('best epoch: {}'.format(best_epoch))


    def _train_step(self, model, data_iterator):

        loss_list = []
        output_list = []
        truth_list = []
        for batch in data_iterator:
            input, label = batch

            self.optimizer.zero_grad()
            logits = model(input.to(torch.device(self.device)))
            if len(logits[0]) == 1:
                label = label.type_as(logits)
                loss = model.loss(logits.squeeze(1), label.to(torch.device(self.device)).view(len(label)))
            else: # Multi-class classification
                loss = model.loss(logits, label.to(torch.device(self.device)).view(len(label)))
            loss_list.append(loss.to(torch.device('cpu')).detach())
            loss.backward()
            self.optimizer.step()
            output_list.append(logits.to(torch.device('cpu')).detach())
            truth_list.append(label.detach())
        loss_train = float(abs(np.mean(loss_list)))
        eval_results = self.evaluate(output_list, truth_list)
        # print('train loss: {}'.format(loss_train))
        print("[{}] {}, loss: {}".format('train', self.print_eval_results(eval_results), loss_train))

        return loss_train, eval_results


    def best_eval_result(self, eval_results):
        assert self.eval_metrics in eval_results, \
            "Evaluation doesn't contain metrics '{}'." \
            .format(self.eval_metrics)

        accuracy = eval_results[self.eval_metrics]
        if accuracy >= self._best_accuracy:
            self._best_accuracy = accuracy
            return True
        else:
            return False


    def _save_log(self, epoch, loss_train, loss_val, eval_results_train, eval_results_val):
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch
        result_each_epoch['loss_train'] = float(loss_train)
        result_each_epoch['accuracy_train'] = float(eval_results_train['accuracy'])
        result_each_epoch['balanced_accuracy_train'] = float(eval_results_train['balanced_accuracy'])
        result_each_epoch['macro_f1_train'] = float(eval_results_train["macro_f1"])
        result_each_epoch['micro_f1_train'] = float(eval_results_train["micro_f1"])
        result_each_epoch['weighted_f1_train'] = float(eval_results_train["weighted_f1"])

        result_each_epoch['loss_validation'] = float(loss_val)
        result_each_epoch['accuracy_validation'] = float(eval_results_val["accuracy"])
        result_each_epoch['balanced_accuracy_validation'] = float(eval_results_val["balanced_accuracy"])
        result_each_epoch['macro_f1_validation'] = float(eval_results_val["macro_f1"])
        result_each_epoch['micro_f1_validation'] = float(eval_results_val["micro_f1"])
        result_each_epoch['weighted_f1_validation'] = float(eval_results_val["weighted_f1"])

        #result_each_epoch['precision_validation'] = float(eval_results_val["precision"])
        #result_each_epoch['recall_validation'] = float(eval_results_val["recall"])
        #result_each_epoch['mcc_validation'] = float(eval_results_val["mcc"])
        #result_each_epoch['AUROC_validation'] = float(eval_results_val["AUROC"])
        #result_each_epoch['AUPR_validation'] = float(eval_results_val["AUPR"])
        #result_each_epoch['TP,TN,FP,FN'] = (int(eval_results_val["TP"]), int(eval_results_val["TN"]), int(eval_results_val["FP"]) ,int(eval_results_val["FN"]))
        self.results[epoch] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log'), 'w') as f:
            json.dump(self.results, f, indent=4)


    def evaluate(self, predict, truth):
        """ Compute evaluation metrics for classification """
        y_trues, y_preds = [], []
        for y_true, logit in zip(truth, predict):
            y_true = y_true.cpu().numpy()
            if len(logit[0]) == 1:
                y_pred = [[np.array([1]) if torch.sigmoid(l).cpu() > 0.5 else np.array([0]) for l in logit]]
            else: # Multi-class classification
                y_pred = [[np.argmax(l).cpu().numpy() for l in logit]]
            y_trues.append(y_true)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0).reshape(len(y_true), 1)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        macro_f1 = metrics.f1_score(y_true, y_pred, pos_label=1, average='macro')
        micro_f1 = metrics.f1_score(y_true, y_pred, pos_label=1, average='micro')
        weighted_f1 = metrics.f1_score(y_true, y_pred, pos_label=1, average='weighted')

        metrics_dict = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": macro_f1,
        "weighted_f1": weighted_f1}

        #confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)

        metrics_dict = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": macro_f1,
        "weighted_f1": weighted_f1}


        # print('y_ture: {}'.format(np.array(y_true).reshape(len(y_true))))
        # print('y_pred: {}'.format(np.array(y_pred).reshape(len(y_pred))))

        return metrics_dict


    def print_eval_results(self, results):
        return ", ".join(
            [str(key) + "=" + "{:.4f}".format(value)
             for key, value in results.items()])



class Tester(object):

    def __init__(self, **kwargs):
        self.save_dir = kwargs['save_dir']
        self.file_list = kwargs['file_list']
        self.device = kwargs['device']
        self.label_list = kwargs['label_list']
        #os.makedirs(os.path.join(self.save_dir, 'figs'), exist_ok=True)

    def test(self, model, data_iter, phase="test"):
        # turn on the testing mode; clean up the history
        model.eval()
        model.phase = phase
        output_list = []
        truth_list = []
        loss_list = []

        # predict
        cnt = 0
        for batch in data_iter:
            input, label = batch
            with torch.no_grad():
                if phase == 'test':
                    try:
                        prediction, feat = model.inference(input.to(torch.device(self.device)))
                        for f, l in zip(feat.to(torch.device('cpu')).detach(), label.detach()):
                            np.save(os.path.join(self.save_dir, 'features',
                                             self.file_list[cnt][:self.file_list[cnt].rfind('.')]), f)
                            cnt += 1
                    except:
                        prediction = model(input.to(torch.device(self.device)))
                else:
                    prediction = model(input.to(torch.device(self.device)))

            output_list.append(prediction.to(torch.device('cpu')).detach())
            truth_list.append(label.detach())
            if len(prediction[0]) == 1:
                label = label.type_as(prediction)
                loss = model.loss(prediction.squeeze(1), label.to(torch.device(self.device)).view(len(label)))
            else: # Multi-class classification
                loss = model.loss(prediction, label.to(torch.device(self.device)).view(len(label)))
            loss_list.append(loss.to(torch.device('cpu')).detach().numpy())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list, phase)
        #auroc, aupr = self.evaluate_auc(output_list, truth_list)
        #eval_results['AUROC'] = auroc
        #eval_results['AUPR'] = aupr

        print("[{}] {}, loss: {}".format(phase, self.print_eval_results(eval_results), abs(np.mean(loss_list))))
        model.phase = 'train'

        return eval_results, abs(np.mean(loss_list))


    def evaluate(self, predict, truth, phase):
        """ Compute evaluation metrics for classification """
        y_trues, y_preds = [], []
        if phase == 'test':
            cnt = 0
            with open(os.path.join(self.save_dir,'predict_list.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['file_name', 'pred', 'label'])
        for y_true, logit in zip(truth, predict):
            y_true = y_true.cpu().numpy()
            if len(logit[0]) == 1:
                y_pred = [[np.array([1]) if torch.sigmoid(l).cpu() > 0.5 else np.array([0]) for l in logit]]
            else: # Multi-class classification
                y_pred = [[np.argmax(l).cpu().numpy() for l in logit]]
            y_trues.append(y_true)
            y_preds.append(y_pred)

            if phase == 'test':
                with open(os.path.join(self.save_dir,'predict_list.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.file_list[cnt], y_pred[0][0], y_true[0][0]])
                    cnt += 1
            
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0).reshape(len(y_true), 1)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        #precision = metrics.precision_score(y_true, y_pred, pos_label=1)
        #recall = metrics.recall_score(y_true, y_pred, pos_label=1)
        macro_f1 = metrics.f1_score(y_true, y_pred, pos_label=1, average='macro')
        micro_f1 = metrics.f1_score(y_true, y_pred, pos_label=1, average='micro')
        weighted_f1 = metrics.f1_score(y_true, y_pred, pos_label=1, average='weighted')

        metrics_dict = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": macro_f1,
        "weighted_f1": weighted_f1}

        # print('y_ture: {}'.format(np.array(y_true).reshape(len(y_true))))
        # print('y_pred: {}'.format(np.array(y_pred).reshape(len(y_pred))))

        if phase == 'test':
            os.makedirs(os.path.join(self.save_dir, 'figs'), exist_ok=True)
            cms = metrics.multilabel_confusion_matrix(y_true, y_pred)
            for cm, l in zip(cms, self.label_list):
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
                disp.plot()
                plt.savefig(os.path.join(self.save_dir, 'figs', 'confusion_matrix_{}.pdf'.format(l)))
            cms = metrics.confusion_matrix(y_true, y_pred)
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cms, display_labels=[str(i) for i in self.label_list])
            disp.plot()
            plt.savefig(os.path.join(self.save_dir, 'figs', 'confusion_matrix_all.pdf'))


        return metrics_dict


    def mcc(self, TP, TN, FP, FN):
         if TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0:
             return 0
         else:
             return ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))


    def evaluate_auc(self, predict, truth):
        """ Compute evaluation metrics for AUC """
        y_trues, y_preds = [], []
        for y_true, logit in zip(truth, predict):
            y_true = y_true.cpu().numpy()
            if len(logit[0]) == 1:
                y_pred = [[torch.sigmoid(l).cpu().numpy() for l in logit]]
            else: # Multi-class classification
                y_pred = [[F.softmax(l, dim=0).cpu().numpy()[1] for l in logit]]
            y_trues.append(y_true)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_trues, axis=0).reshape(len(y_trues))
        y_pred = np.concatenate(y_preds, axis=0).reshape(len(y_preds))

        auroc = self._vis_auroc(y_pred, y_true)
        aupr = self._vis_aupr(y_pred, y_true)
        np.savez(os.path.join(self.save_dir, 'log_auc.npz'), y_pred=y_pred, y_true=y_true)

        return auroc, aupr


    def _vis_auroc(self, y_pred, y_true):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auroc = metrics.roc_auc_score(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, marker='o', label='AUROC={:.4f}'.format(auroc))
        plt.xlabel('False positive rate', fontsize=18)
        plt.ylabel('True positive rate', fontsize=18)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc=4, fontsize=18)
        plt.savefig(os.path.join(self.save_dir, 'figs', 'AUROC.pdf'))
        plt.close()
        return auroc

    def _vis_aupr(self, y_pred, y_true):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        aupr = metrics.average_precision_score(y_true, y_pred)
        precision = np.concatenate([np.array([0.0]), precision])
        recall = np.concatenate([np.array([1.0]), recall])
        plt.figure(figsize=(8, 8))
        plt.plot(precision, recall, marker='o', label='AUPR={:.4f}'.format(aupr))
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc=3, fontsize=18)
        plt.savefig(os.path.join(self.save_dir, 'figs', 'AUPR.pdf'))
        plt.close()
        return aupr

    def vis_attn_weights(self, attn_weights_list, predict, truth):
        y_trues, y_preds = [], []
        cnt = 0
        for y_true, logit, aw in zip(truth, predict, attn_weights_list):
            y_true = y_true.cpu().numpy()
            if len(logit[0]) == 1:
                y_pred = [[np.array([1]) if torch.sigmoid(l).cpu() > 0.5 else np.array([0]) for l in logit]]
            else: # Multi-class classification
                y_pred = [[np.argmax(l).cpu().numpy() for l in logit]]
            #aw = aw.squeeze(0).cpu().numpy()
            aw = aw.squeeze(0)
            filename = os.path.join(self.save_dir, 'figs', 'attention_weight_{}.pdf'.format(self.file_list[cnt]))
            np.savez(os.path.join(self.save_dir, 'attention_weights', 'attn_weight_{}.npz'.format(self.file_list[cnt])), arr_0=aw)
            self.make_heatmap(np.flipud(aw), y_pred, y_true, filename)
            cnt += 1


    def print_eval_results(self, results):
        return ", ".join(
            [str(key) + "=" + "{:.4f}".format(value)
             for key, value in results.items()])
