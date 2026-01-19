# Copyright (c) 2022-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Manage row actions from Python.

* Subclass `Workflow` to create a new workflow.
* Call ``YourWorkflow.add_action`` to add a new action to the workflow.
* Call ``YourWorkflow.write_workflow`` to write ``workflow.toml`` with the configuration
  of all actions.
* Call ``YourWorkflow.main()`` in ``project.py`` to parse command line arguments and
  dispatch the correct action method.
"""

import argparse
import subprocess
import warnings
from pathlib import Path

import rtoml
import signac


def _get_cluster_name():
    """Get the current cluster name."""
    result = subprocess.run(
        ['row', 'show', 'cluster', '--short'],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


class Action:
    """Represent a row action.

    An `Action` consists of a method that implements the action and the row
    configuration options for that action. The method is called by `__call__`. The
    configuration is stored as a raw dictionary that maps directly to the ``[action]``
    element of the row ``workflow.toml``.

    The method must be a function that takes the argument(s) ``*jobs``.

    Args:
        method(callable): The method that implements this action. It must take the
            argument ``*jobs``.
        configuration(dict): Configuration options for the action to be written to
            ``workflow.toml``.
    """

    def __init__(self, method, configuration):
        if 'name' in configuration:
            message = 'configuration must not contain "name"'
            raise ValueError(message)

        self._method = method
        self._configuration = configuration

    def __call__(self, *jobs):
        """Call the `method` given on construction."""
        self._method(*jobs)


class Workflow:
    """Represent a single workflow."""

    _actions = {}

    @classmethod
    def add_action(cls, name, action):
        """Add an action.

        Args:
            name(str): The action's name. Must be unique.
            action(Action): The action itself.
        """
        if name in cls._actions:
            message = f'Action {name} cannot be added twice.'
            raise ValueError(message)

        if 'products' not in action._configuration:
            warnings.warn(f'Action {name} is missing products.', stacklevel=2)

        cls._actions[name] = action

    @classmethod
    def write_workflow(cls, entrypoint, path=None, default=None, account=None):
        """Write the file ``workflow.toml``.

        ``workflow.toml`` will include the signac workspace definition, the given
        ``default`` mapping (when provided), and configurations for all added actions.

        Note:
            ``default.action.command`` will be automatically set based on the value of
            `entrypoint`.

        Args:
            entrypoint(str): Name of the python file that calls the `main` entrypoint.
            path(Path): Path to write ``workflow.toml``.
            default(dict): The ``[default]`` mapping.
            account(str): Name of the cluster account to use.
        """
        workflow = {
            'workspace': {'path': 'workspace', 'value_file': 'signac_statepoint.json'}
        }

        workflow['default'] = {
            'action': {
                'command': f'python -u {entrypoint} action $ACTION_NAME {{directories}}'
            }
        }
        if account is not None:
            workflow['default']['action'].update(
                {'submit_options': {_get_cluster_name(): {'account': account}}}
            )

        if default is not None:
            workflow['default'].update(default)

        workflow['action'] = []
        for name, action_item in cls._actions.items():
            action = {'name': name}
            action.update(action_item._configuration)
            workflow['action'].append(action)

        if path is None:
            path = Path('.')

        with open(path / 'workflow.toml', 'w', encoding='utf-8') as workflow_file:
            rtoml.dump(workflow, workflow_file, pretty=True)

    @classmethod
    def main(cls, init=None, init_args=None, **kwargs):
        """Implement the main entrypoint for ``project.py``.

        Valid commands are:
        * ``python project.py init``
        * ``python project.py action action_name directories``

        ``init`` will call the user-provided method ``init``, then generate the file
        ``workflow.toml``. When provided, items in the ``init_args`` list will be added
        as options to the ``init`` subparser with ``add_argument``.

        Args:
            init(callable): User-provided initialization routine. Must take one
                argument: ``args`` - the ``argparse`` parsed arguments.
            init_args(list[str]): List of args to add to the ``init`` subparser.
            **kwargs: Forwarded to `make_workflow`.
        """
        parser = argparse.ArgumentParser()
        command = parser.add_subparsers(dest='command', required=True)
        init_parser = command.add_parser('init')
        init_parser.add_argument('--account')
        if init_args is not None:
            for arg in init_args:
                init_parser.add_argument(arg)

        action_parser = command.add_parser('action')
        action_parser.add_argument('action')
        action_parser.add_argument('directories', nargs='+')

        args = parser.parse_args()

        if args.command == 'init':
            if init is not None:
                init(args)

            cls.write_workflow(account=args.account, **kwargs)
        elif args.command == 'action':
            project = signac.get_project()
            jobs = [project.open_job(id=directory) for directory in args.directories]
            cls._actions[args.action](*jobs)

        else:
            message = f'Invalid command: {args.command}'
            raise RuntimeError(message)
