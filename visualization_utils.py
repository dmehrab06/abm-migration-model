# trajectory_tracker.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from numpy import dot
from numpy.linalg import norm
from file_paths_and_consts import *
import os
import subprocess
import ast
import math
from matplotlib import rcParams
import matplotlib.dates as mdates
import json
from result_utils import *
from matplotlib.patches import FancyBboxPatch

SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_simulation_trajectory(seed):
    
    if seed==199:
        return [199,199,214,214,236,236,236,236,270,282,295,301,319,320,339,339,339,339,339,339]
    
    cmd = [
        'python',
        'parse_calibration_log.py',
        '--seed',f'{seed}',
    ]

    #print(f"Command: {' '.join(cmd)}")

    result = (subprocess.run(cmd,capture_output=True,text=True)).stdout.splitlines()
    parsed = ast.literal_eval(result[-1].replace('nan', 'None'))[0]  # Get the inner list
    parsed = [seed] + parsed
    #print(parsed)
    valid_numbers = pd.Series(parsed).ffill().tolist()
    #print(valid_numbers)
    return valid_numbers

def plot_simulations(epoch,selected_simulations,pnas_refugee_data,ax=None,optimize=True,train_only=True,savefig=None,
                    ground_truth_upto='2022-05-15',simulation_upto='2022-05-15'):
    
    simulation_color = '#084594'
    ground_color = '#005a32'
    
    lgd_params = {'hlen':1,'bpad':0.2,'lspace':0.2,'htxtpad':0.2,'baxpad':0.2,'cspace':0.2,'ncol':1,'ecolor':'black','size':28,'alpha':0.2}
    
    tight = False
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,5))
        tight=True
    for simidx, sim in enumerate(selected_simulations):
        all_dfs = get_result_for_sim(int(sim))
        daily_total_refugee_df = get_daily_total_refugee_for_sim(all_dfs,upto=simulation_upto)
        if optimize:
            best_scale = optimize_scale(daily_total_refugee_df,pnas_refugee_data,upto=simulation_upto)
            daily_total_refugee_df['total_refugee'] = daily_total_refugee_df['total_refugee']*best_scale
        nrmse,_,corr,_,_ = get_metrics_of_daily_sim(daily_total_refugee_df,pnas_refugee_data,upto=simulation_upto)
        print(nrmse,corr)
        #print(best_scale)
        #if nrmse<0.2 and corr>0.87:
        ax.plot(daily_total_refugee_df['date'],(daily_total_refugee_df['total_refugee'])/(1000),color=simulation_color,
                label=f'Epoch {epoch}',alpha=1.0 if train_only else 1.0)
        #config = get_config_for_sim(sim)
        #print(sim,nrmse,corr)
        #show_params(config)
        #print('')
    
    ax.plot(pnas_refugee_data['time'],pnas_refugee_data['daily']/1000,linestyle='',marker='o',
            markevery=2,markeredgecolor=ground_color,markerfacecolor='white',markersize=5)
    myFmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlim([pd.to_datetime('2022-03-02'),pd.to_datetime(ground_truth_upto)])
    ax.set_xlabel('Date')
    ax.set_ylabel('Refugee (K)')
    
    if train_only:
        ax.axvline(x=pd.to_datetime(simulation_upto),label='Training Period',linestyle='--',color='black')
    ax.legend(loc="best", fancybox=True, handlelength=lgd_params['hlen'], borderpad=lgd_params['bpad'], labelspacing=lgd_params['lspace'],
                  handletextpad = lgd_params['htxtpad'], borderaxespad = lgd_params['baxpad'], columnspacing = lgd_params['cspace'],
                  ncol=lgd_params['ncol'], edgecolor=lgd_params['ecolor'], frameon=True, framealpha=lgd_params['alpha'], shadow=False)
        
    #if savefig is not None:
    #    fig.savefig(f'time_series_figs/{savefig}',dpi=350,bbox_inches='tight')

class RadarChartGenerator:
    def __init__(self, config_dir='CoordinateDescentconfig', output_dir='radar_charts',figsize=(8,8)):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        
        # Parameter names and display labels
        self.param_names = [
            "param_migration_bias",
            "param_distance_decay", 
            "param_discount_rate",
            "param_risk_growth_rate",
            "param_threshold_hi",
            "param_lambda1",
            "param_lambda2",
            "refugee_among_displaced"
        ]
        
        self.param_labels = [
            "Migration\nBias",
            "Distance\nDecay",
            "Discount\nRate",
            "Risk Growth\nRate",
            "Threshold\nHi",
            "Lambda 1",
            "Lambda 2",
            "Refugee\nRatio"
        ]
        
        self.param_labels_math = [
            "$Q$",
            "$\\delta$",
            "$\\theta$",
            "$v$",
            "$\\tau$",
            "$\\gamma_1$",
            "$\\gamma_2$",
            "$\\mathcal{P}$"
        ]
        
        # Parameter descriptions for the legend box
        self.param_descriptions = {
            "$Q$": "Bias Control (Sets baseline propensity to migrate)",
            "$\\delta$": "Distance decay (Spatial kernel parameter)",
            "$\\theta$": "Discount Factor (Temporal kernel parameter)",
            "$v$": "Risk growth rate (Controls migration propensity with growth of event)",
            "$\\tau$": "Influence threshold (decision boundary)",
            "$\\gamma_1$": "Social influence weight (Controls influence from peer)",
            "$\\gamma_2$": "Cognitive influence weight (Controls influence from events)",
            "$\\mathcal{P}$": "Refugee ratio (proportion of border crossing)"
        }
        
        # Parameter bounds for normalization
        self.param_bounds = {
            "param_migration_bias": (0.0, 300.0),
            "param_distance_decay": (1.0, 10.0),
            "param_discount_rate": (0.0, 1.0),
            "param_risk_growth_rate": (0.0, 50.0),
            "param_threshold_hi": (0.0, 1.0),
            "param_lambda1": (0.0, 1.0),
            "param_lambda2": (0.0, 1.0),
            "refugee_among_displaced": (0.0, 1.0)
        }
        
        self.chart_angles = [0.0,np.pi/3.0,(2*np.pi)/3.0,np.pi,np.pi+np.pi/5,np.pi+(2*np.pi)/5,np.pi+(3*np.pi)/5,np.pi+(4*np.pi)/5]
        
        self.ground_truth_data = pd.read_csv('/project/biocomplexity/UKR_forecast/migration_data/gtruth_from_various_source/ukr_refugee_src_pnas.csv')
        self.ground_truth_data['time'] = pd.to_datetime(self.ground_truth_data['time'])
        self.ground_truth_data['daily'] = self.ground_truth_data['daily'].rolling(7).mean()
        self.ground_truth_data = (self.ground_truth_data).dropna(subset=['daily'])
    
    def normalize_value(self, param_name, value):
        """Normalize parameter value to [0, 1] range for radar chart"""
        min_val, max_val = self.param_bounds[param_name]
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    
    def load_epoch_config(self, seed, epoch):
        """Load configuration for a specific epoch"""
        config_file = self.config_dir / f"config_coord_epoch_{epoch}_from_{seed}.json"
        if not config_file.exists():
            return None
        
        with open(config_file) as f:
            config = json.load(f)
            config['params']['param_discount_rate'] = config['params']['param_discount_rate']/100.0
            return config
    
    def load_summary(self, seed, epoch):
        """Load summary for a specific epoch"""
        summary_file = self.config_dir / f"summary_seed_{seed}_epoch_{epoch}.json"
        if not summary_file.exists():
            return None
        
        with open(summary_file) as f:
            return json.load(f)
    
    def draw_parameter_legend(self, ax):
        """
        Draw a clean parameter description box below the radar chart
        
        Args:
            ax: Matplotlib axis to draw on
        """
        # Hide axis
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Parameter Descriptions', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        # Create two columns for better space utilization
        left_params = list(self.param_descriptions.items())[:4]
        right_params = list(self.param_descriptions.items())[4:]
        
        y_start = 0.82
        y_step = 0.2
        x_left = 0.04
        x_right = 0.52
        
        # Left column
        for i, (symbol, description) in enumerate(left_params):
            y_pos = y_start - i * y_step
            
            # Parameter symbol (bold, larger)
            ax.text(x_left, y_pos, symbol, 
                   ha='left', va='top', fontsize=13, fontweight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='#e3f2fd', 
                           edgecolor='#1976d2',
                           linewidth=1.5))
            
            # Description
            ax.text(x_left + 0.02, y_pos - 0.01, description, 
                   ha='left', va='top', fontsize=10,
                   transform=ax.transAxes, wrap=True)
        
        # Right column
        for i, (symbol, description) in enumerate(right_params):
            y_pos = y_start - i * y_step
            
            # Parameter symbol (bold, larger)
            ax.text(x_right, y_pos, symbol, 
                   ha='left', va='top', fontsize=13, fontweight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='#e3f2fd', 
                           edgecolor='#1976d2',
                           linewidth=1.5))
            
            # Description
            ax.text(x_right + 0.02, y_pos - 0.01, description, 
                   ha='left', va='top', fontsize=10,
                   transform=ax.transAxes, wrap=True)
        
        # Add a subtle border around the legend box
        
        box = FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                            boxstyle="round,pad=0.01",
                            transform=ax.transAxes,
                            facecolor='#fafafa',
                            edgecolor='#9e9e9e',
                            linewidth=1.5,
                            zorder=0)
        ax.add_patch(box)
    
    def draw_radar_chart(self,seed,epoch,values, metrics_values, metrics_labels, output_file):
        fig = plt.figure(figsize=self.figsize)  # slightly smaller aspect ratio helps balance
        # gs = GridSpec(
        #     1, 3, figure=fig,
        #     width_ratios=[3, 1, 2],   # radar : bar chart ratio
        #     wspace=0.05,
        #     left=0.05, right=0.95,
        #     top=0.9, bottom=0.1
        # )
        
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1.5, 1, 0.8], 
                      wspace=0.1, hspace=0.15,left=0.05, right=0.95,top=0.95, bottom=0.1)

        # Subplots without manual positions
        ax_radar = fig.add_subplot(gs[0:2, 0], projection='polar')
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_refugee = fig.add_subplot(gs[1, 1])
        ax_legend = fig.add_subplot(gs[2, :])
        # --------------------
        # Radar chart styling
        # --------------------
        N = len(values)
        angles = (self.chart_angles).copy()
        values += values[:1]
        angles += angles[:1]
        #print('length of values',len(values))
        #print('length of angles',len(angles))
        ax_radar.plot(angles, values, linewidth=2, linestyle='solid', color='#2a7fba')
        #ax_radar.fill(angles, values, color='#2a7fba', alpha=0.25)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_title(f'Seed {seed if seed!=199 else 84} Epoch {epoch}')
        # ax_radar.set_yticklabels(['20%', '40%', '60%', '80%', '100%'],fontsize=9, color='gray')
        # ax_radar.tick_params(axis='y', pad=10)
        
        for angle, value, param_name, label in zip(angles[:-1], values, self.param_names, self.param_labels):
            x_pos = angle
            y_pos = value
            
            actual_value = value*(self.param_bounds[param_name][1]-self.param_bounds[param_name][0])+self.param_bounds[param_name][0]
            
            value_str = f'{actual_value:.3f}' if actual_value < 0.01 else f'{actual_value:.1f}'

            ax_radar.text(x_pos, y_pos, value_str, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12, color='#2E86AB',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor='#2E86AB',
                           alpha=1.0))
            
            max_value_str = f'{self.param_bounds[param_name][1]:.1f}'

            ax_radar.text(x_pos+np.pi/50, 0.95, max_value_str, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=8, color='black',rotation=90,
                   bbox=dict(boxstyle='round,pad=0.1', 
                           facecolor='white', 
                           edgecolor='black',
                           alpha=0.2))

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(self.param_labels_math,fontsize=18)
        ax_radar.set_yticklabels([])
        ax_radar.grid(alpha=0.5)

        # Keep radar chart circular (important!)
        ax_radar.set_aspect('equal', adjustable='box')

        # --------------------
        # Metrics bar chart
        # --------------------
        bars = ax_metrics.bar(metrics_labels, metrics_values, color=['#e57373', '#4db6ac'])
        for bar, val in zip(bars, metrics_values):
            ax_metrics.text(
                bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5)
            )
        ax_metrics.set_ylim(0, 1)
        ax_metrics.set_title('Objectives')
        #ax_metrics.set_ylabel("Value", fontsize=12, fontweight='bold')
        
        # --------------------
        # simulation time series plot
        # --------------------
        plot_simulations(epoch,[self.simulation_trajectory[epoch]],self.ground_truth_data,ax=ax_refugee,
                         simulation_upto='2022-03-15',optimize=True,train_only=True)
        # --------------------
        # Save
        # --------------------
        
        self.draw_parameter_legend(ax_legend)
        
        plt.savefig(output_file, dpi=300, pad_inches=0.05, facecolor='white')
        plt.close(fig)
        print(f"Saved: {output_file}")
        return ax_radar,ax_metrics,ax_refugee

    
    def create_radar_chart(self, seed, epoch, config, summary=None, save=True):
        """
        Create a radar chart for a specific epoch with metrics subplot

        Args:
            seed: Calibration seed
            epoch: Epoch number
            config: Configuration dict with parameter values
            summary: Optional summary dict with metrics
            save: Whether to save the figure
        """
        # Extract parameter values
        params = config['params']
        values = [params[p] for p in self.param_names]

        # Normalize values to [0, 1]
        normalized_values = [self.normalize_value(p, v) 
                            for p, v in zip(self.param_names, values)]

        # Number of variables
        num_vars = len(self.param_names)

        # Compute angle for each axis
        angles = (self.chart_angles).copy()
        
        #print('has summary')
        nrmse = summary.get('best_nrmse', 0)
        pcc = summary.get('best_pcc', 0)

        # Create vertical bars
        metrics = ['NRMSE', 'PCC']
        values_metrics = [nrmse, pcc]
        output_file = self.output_dir / f'radar_seed{seed}_epoch{epoch:02d}.png'
        #plt.savefig(output_file, dpi=350, facecolor='white', edgecolor='none')
        #print(f"Saved: {output_file}")
        return self.draw_radar_chart(seed,epoch,normalized_values,values_metrics,['NRMSE','PCC'],output_file)
    
    def generate_all_epochs(self, seed, max_epoch=None,figsize=(10,10)):
        """
        Generate radar charts for all available epochs
        
        Args:
            seed: Calibration seed
            max_epoch: Maximum epoch to generate (None = all available)
        """
        
        self.simulation_trajectory = get_simulation_trajectory(seed)
        
        epoch = 0
        charts_generated = []
        
        while True:
            #print('epoch',epoch)
            if max_epoch is not None and epoch > max_epoch:
                break
            
            config = self.load_epoch_config(seed, epoch)
            if config is None:
                break
            
            summary = self.load_summary(seed, epoch)
            
            if summary is not None:
                ax_radar, ax_metric, ax_refugee = self.create_radar_chart(seed, epoch, config, summary)
                #plt.close(fig)
            
            charts_generated.append(epoch)
            epoch += 1
        
        print(f"\nGenerated {len(charts_generated)} radar charts for seed {seed}")
        print(f"Epochs: {charts_generated}")
        return charts_generated
    
    def compare_params_at_epoch(self, seed_1, seed_2,max_epoch=20):
        """
        Compare parameters of two epochs based on cosine similarity
        """
        epoch = 0
        
        while True:
            #print('epoch',epoch)
            if max_epoch is not None and epoch > max_epoch:
                break
            
            config1 = self.load_epoch_config(seed_1, epoch)
            param1 = np.array([config1['params'][p] for p in self.param_names])
            
            config2 = self.load_epoch_config(seed_2, epoch)
            param2 = np.array([config2['params'][p] for p in self.param_names])
            
            cos_sim = dot(param1, param2)/(norm(param1)*norm(param2))
            print(f'At epoch {epoch} cosine similarity is {cos_sim}')
            
            epoch += 1