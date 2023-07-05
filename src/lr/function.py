import os
import torch
import numpy as np
import torch.nn as nn
from .model import *

"""線形回帰モデル"""
class LR(object):
    def __init__(self, i_dim, o_dim, mode=False, model_path=''):
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル定義
        self.__model=Model(i_dim, o_dim).to(device=self.__device)

        # 学習済みモデル
        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))
        
        # 学習係数
        self.__lr=1e-3
        # 損失関数:最小二乗法
        self.__loss_func=nn.MSELoss()
        # 最適化アルゴリズム:SGD
        self.__opt=torch.optim.SGD(self.__model.parameters(), lr=self.__lr)

        # save file path
        self.FILE_PATH=os.path.join('./model')

        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)

    """fit:フィッティング処理"""
    def fit(self, X, Y, mode=False):
        # 損失を格納変数
        losses=torch.zeros(len(X))
        X=X.to(device=self.__device)
        Y=Y.to(device=self.__device)

        for e in range(100):
            # 予測
            y_hat=self.__model(X)
            # 損失計算
            loss=self.__loss_func(y_hat, Y)
            
            # 勾配を初期化
            self.__opt.zero_grad()
            
            # 逆伝播を計算
            loss.backward()
            
            # 次のステップ
            self.__opt.step()

            # 損失を格納
            losses[e]=loss.item()

        # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'loss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, losses)
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'parameter.txt')
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
        
    """pred:予測処理"""
    def pred(self, x):
        return self.__model(x)
    
    """get_params:モデルパラメータを呼び出す"""
    def get_params(self):
        [w, b]=self.__model.parameters()
        return (w[0][0].item(), b[0].item())
    