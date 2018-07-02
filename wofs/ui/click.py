import click
import logging
import os
import copy


def _print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    click.echo(
        '{prog}, version {version}'.format(
            prog='Open Data Cube core',
            version=__version__
        )
    )
    ctx.exit()


class ColorFormatter(logging.Formatter):
    colors = {
        'info': dict(fg='white'),
        'error': dict(fg='red'),
        'exception': dict(fg='red'),
        'critical': dict(fg='red'),
        'debug': dict(fg='blue'),
        'warning': dict(fg='yellow')
    }

    def format(self, record):
        if not record.exc_info:
            record = copy.copy(record)
            record.levelname = click.style(record.levelname, **self.colors.get(record.levelname.lower(), {}))
        return logging.Formatter.format(self, record)


class ClickHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            click.echo(msg, err=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # pylint: disable=bare-except
            self.handleError(record)


def remove_handlers_of_type(logger, handler_type):
    for handler in logger.handlers:
        if isinstance(handler, handler_type):
            logger.removeHandler(handler)


def _init_logging(ctx, param, value):
    # When running in tests, we don't want to keep adding log handlers. It creates duplicate log messages up the wahoo.
    remove_handlers_of_type(logging.root, ClickHandler)
    handler = ClickHandler()
    handler.formatter = ColorFormatter(_LOG_FORMAT_STRING)
    logging.root.addHandler(handler)

    logging_level = logging.WARN - 10 * value
    logging.root.setLevel(logging_level)
    logging.getLogger('datacube').setLevel(logging_level)

    if logging_level <= logging.INFO:
        logging.getLogger('rasterio').setLevel(logging.INFO)

    logging.getLogger('datacube').info('Running datacube command: %s', ' '.join(sys.argv))

    if not ctx.obj:
        ctx.obj = {}

    ctx.obj['verbosity'] = value


def _log_queries(ctx, param, value):
    if value:
        logging.getLogger('sqlalchemy.engine').setLevel('INFO')


def _set_config(ctx, param, value):
    if value:
        if not any(os.path.exists(p) for p in value):
            raise ValueError('No specified config paths exist: {}' % value)

        if not ctx.obj:
            ctx.obj = {}
        paths = value
        ctx.obj['config_files'] = paths


#: pylint: disable=invalid-name
version_option = click.option('--version', is_flag=True, callback=_print_version,
                              expose_value=False, is_eager=True)
#: pylint: disable=invalid-name
verbose_option = click.option('--verbose', '-v', count=True, callback=_init_logging,
                              is_eager=True, expose_value=False, help="Use multiple times for more verbosity")
#: pylint: disable=invalid-name
config_option = click.option('--config', '--config_file', '-C', multiple=True, default='', callback=_set_config,
                             expose_value=False)
#: pylint: disable=invalid-name
log_queries_option = click.option('--log-queries', is_flag=True, callback=_log_queries,
                                  expose_value=False, help="Print database queries.")