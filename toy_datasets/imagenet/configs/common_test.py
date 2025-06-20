"""Tests for config utils."""

from absl.testing import parameterized
from google3.experimental.largesensormodels.toy_datasets.imagenet.configs import common

from google3.testing.pybase import googletest


class CommonTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_parse_arg_works(self, lazy):
    spec = dict(
        res=224,
        lr=0.1,
        runlocal=False,
        schedule='short',
    )

    def check(result, runlocal, schedule, res, lr):
      self.assertEqual(result.runlocal, runlocal)
      self.assertEqual(result.schedule, schedule)
      self.assertEqual(result.res, res)
      self.assertEqual(result.lr, lr)
      self.assertIsInstance(result.runlocal, bool)
      self.assertIsInstance(result.schedule, str)
      self.assertIsInstance(result.res, int)
      self.assertIsInstance(result.lr, float)

    check(common.parse_arg(None, lazy=lazy, **spec), False, 'short', 224, 0.1)
    check(common.parse_arg('', lazy=lazy, **spec), False, 'short', 224, 0.1)
    check(common.parse_arg('runlocal=True', lazy=lazy, **spec), True, 'short',
          224, 0.1)
    check(common.parse_arg('runlocal=False', lazy=lazy, **spec), False, 'short',
          224, 0.1)
    check(common.parse_arg('runlocal=', lazy=lazy, **spec), False, 'short', 224,
          0.1)
    check(common.parse_arg('runlocal', lazy=lazy, **spec), True, 'short', 224,
          0.1)
    check(common.parse_arg('res=128', lazy=lazy, **spec), False, 'short', 128,
          0.1)
    check(common.parse_arg('128', lazy=lazy, **spec), False, 'short', 128, 0.1)
    check(common.parse_arg('schedule=long', lazy=lazy, **spec), False, 'long',
          224, 0.1)
    check(common.parse_arg('runlocal,schedule=long,res=128', lazy=lazy, **spec),
          True, 'long', 128, 0.1)

  @parameterized.parameters(
      (None, {}, {}),
      (None, {'res': 224}, {'res': 224}),
      ('640', {'res': 224}, {'res': 640}),
      ('runlocal', {}, {'runlocal': True}),
      ('res=640,lr=0.1,runlocal=false,schedule=long', {},
       {'res': 640, 'lr': 0.1, 'runlocal': False, 'schedule': 'long'}),
      )
  def test_lazy_parse_arg_works(self, arg, spec, expected):
    self.assertEqual(dict(common.parse_arg(arg, lazy=True, **spec)), expected)


if __name__ == '__main__':
  googletest.main()
