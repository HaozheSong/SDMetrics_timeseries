"""Timeseries quality report"""
import os
import sys
import pickle
import random
import warnings
import pkg_resources
import dash
import inspect
import importlib
import pprint


from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm import tqdm
from config_io import Config
from collections import OrderedDict


class QualityReport:
    def __init__(self, config_file=None, config_dict=None):
        if config_dict is not None:
            self._config = Config(config_dict)
        else:
            self._config = Config.load_from_file(
                config_file, default="config_quality_report.json",
                default_search_paths=[os.path.dirname(
                    inspect.getfile(self.__class__))],)
        self.graph_idx = 0

    # Different metrics have different depths
    # E.g., `single_attr_dist` has depth=2, `interarrival` has depth=1
    # TODO: prettier layout
    def _traverse_metrics_dict(self, metrics_dict, html_children):
        for main_metric, scores in metrics_dict.items():
            html_children.append(html.Div(html.H2(main_metric)))
            if isinstance(scores, list):
                score = scores[0]
                if len(scores) > 1 and scores[1] is not None:  # valid plot
                    fig = scores[1]
                    html_children.append(
                        html.Div(
                            [
                                html.Div(
                                    "score: {:.3f} (best: {:.3f}, worst: {:.3f})".format(
                                        score[0], score[1], score[2]
                                    )
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            # TODO: better graph index
                                            id=f"graph-{self.graph_idx}",
                                            figure=fig,
                                            style={"width": "100vh"},
                                        )
                                    ]
                                ),
                            ]
                        )
                    )
                    self.graph_idx += 1
                else:
                    html_children.append(
                        html.Div(
                            "score: {:.3f} (best: {:.3f}, worst: {:.3f})".format(
                                score[0], score[1], score[2]
                            )
                        )
                    )
            else:
                self._traverse_metrics_dict(scores, html_children)

    def visualize(self):
        app = dash.Dash(__name__)
        html_children = []
        for metric_type, metrics_dict in self.dict_metric_scores.items():
            html_children.append(
                html.Div(
                    html.H1(children=metric_type),
                )
            )
            self._traverse_metrics_dict(metrics_dict, html_children)

        app.layout = html.Div(children=html_children)
        app.run_server(debug=False, host='0.0.0.0')

    def serialize(self):
        dict_metric_scores = self.dict_metric_scores
        # ------------------------------depth=2------------------------------
        # dict_metric_scores = {
        #     'fidelity': {
        #         'Single attribute distributional similarity': {
        #             'srcip': [(score, best, worst), Figure],
        #             'dstip': [(0.36005828477184515, 0.0, 1.0), Figure],
        #         },
        #         'Single feature distributional similarity': {
        #             'td': [(score, best, worst), Figure],
        #             'pkt': [(1806.4343452780117, 0.0, inf), Figure],
        #         }
        #     }
        # }
        # ------------------------------depth=1------------------------------
        # dict_metric_scores = {
        #     'fidelity': {
        #         'srcip': [(score, best, worst), Figure],
        #         'dstip': [(0.36005828477184515, 0.0, 1.0), Figure],
        #         'pkt': [(1806.4343452780117, 0.0, inf), Figure],
        #     }
        # }
        return

    def get_fig_refs(self, dict_var, figs_dict):
        for key, value in dict_var.items():
            if not isinstance(value, list):
                self.get_fig_refs(value, figs_dict)
            else:
                figs_dict[key] = value[1]

    def fig2png(self, save_folder):
        figs_dict = {}
        self.get_fig_refs(self.dict_metric_scores, figs_dict)
        os.makedirs(save_folder, exist_ok=True)
        for fig_name, fig_obj in figs_dict.items():
            img_bytes = fig_obj.to_image(format='png')
            save_path = os.path.join(save_folder, fig_name + '.png')
            with open(save_path, 'wb') as img_file:
                img_file.write(img_bytes)

    def fig2html(self, save_folder, full_html=False):
        figs_dict = {}
        self.get_fig_refs(self.dict_metric_scores, figs_dict)
        os.makedirs(save_folder, exist_ok=True)
        for fig_name, fig_obj in figs_dict.items():
            html_str = fig_obj.to_html(full_html=full_html)
            save_path = os.path.join(save_folder, fig_name + '.html')
            with open(save_path, 'w') as html_file:
                html_file.write(html_str)

    def fig2json(self, save_folder, pretty=True, remove_uids=False):
        figs_dict = {}
        self.get_fig_refs(self.dict_metric_scores, figs_dict)
        os.makedirs(save_folder, exist_ok=True)
        for fig_name, fig_obj in figs_dict.items():
            json_str = fig_obj.to_json(pretty=pretty, remove_uids=remove_uids)
            save_path = os.path.join(save_folder, fig_name + '.json')
            with open(save_path, 'w') as json_file:
                json_file.write(json_str)

    def generate(self, real_data, synthetic_data, metadata, out=sys.stdout):
        self.dict_metric_scores = OrderedDict()

        for metric_type, metrics in self._config["metrics"].items():
            # fidelity/privacy
            metric_module = importlib.import_module(
                f"sdmetrics.timeseries.{metric_type}"
            )
            self.dict_metric_scores[metric_type] = OrderedDict()
            for metric_dict in metrics:
                metric_name = list(metric_dict.keys())[0]
                metric_config = list(metric_dict.values())[0]
                metric_class = getattr(metric_module, metric_config["class"])
                metric_class_instance = metric_class()

                # Metrics that do not have `target` (e.g., session length)
                if "target_list" not in metric_config:
                    _real_data = real_data.copy(deep=True)
                    _synthetic_data = synthetic_data.copy(deep=True)
                    self.dict_metric_scores[metric_type][
                        metric_name
                    ] = metric_class_instance._insert_best_worst_score_metrics_output(
                        metric_class_instance.compute(
                            _real_data, _synthetic_data, metadata,
                            configs=getattr(metric_config, "configs", None)
                        )
                    )

                # Metrics that have `target` (e.g., single attribute distributional similarity)
                else:
                    if metric_name not in self.dict_metric_scores[metric_type]:
                        self.dict_metric_scores[metric_type][metric_name] = OrderedDict(
                        )
                    for target in metric_config["target_list"]:
                        _real_data = real_data.copy(deep=True)
                        _synthetic_data = synthetic_data.copy(deep=True)
                        self.dict_metric_scores[metric_type][metric_name][
                            str(target)
                        ] = metric_class_instance._insert_best_worst_score_metrics_output(
                            metric_class_instance.compute(
                                _real_data, _synthetic_data, metadata, target=target,
                                configs=getattr(metric_config, "configs", None)
                            )
                        )

    def save(self, filepath):
        """Save this report instance to the given path using pickle.

        Args:
            filepath (str):
                The path to the file where the report instance will be serialized.
        """
        self._package_version = pkg_resources.get_distribution(
            "sdmetrics").version

        with open(filepath, "wb") as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a ``QualityReport`` instance from a given path.

        Args:
            filepath (str):
                The path to the file where the report is stored.

        Returns:
            QualityReort:
                The loaded quality report instance.
        """
        current_version = pkg_resources.get_distribution("sdmetrics").version

        with open(filepath, "rb") as f:
            report = pickle.load(f)
            if current_version != report._package_version:
                warnings.warn(
                    f"The report was created using SDMetrics version `{report._package_version}` "
                    f"but you are currently using version `{current_version}`. "
                    "Some features may not work as intended.")

            return report
