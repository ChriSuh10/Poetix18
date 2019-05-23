using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rhetorica
{
  public static class Miscellaneous
  {
    #region "Console-App Methods"
    public static void Pause(string message)
    {
      if (message == string.Empty)
        message = "Press any key to continue.";
      Console.WriteLine(message);
      Console.ReadKey();
    }

    public static void PauseLine(string message)
    {
      if (message == string.Empty)
        message = "Press [Enter] to continue.";
      Console.WriteLine(message);
      Console.ReadLine();
    }
    #endregion

    #region "Algorithms"
    // V. http://stackoverflow.com/questions/1952153/what-is-the-best-way-to-find-all-combinations-of-items-in-an-array/10629938
    public static IEnumerable<IEnumerable<T>> GetPermutations<T>(IEnumerable<T> list, int length)
    {
      if (length == 1) return list.Select(t => new T[] { t });

      return GetPermutations(list, length - 1)
        .SelectMany(t => list.Where(e => !t.Contains(e)),
          (t1, t2) => t1.Concat(new T[] { t2 }));
    }


    public static List<List<T>> GetPermutationsList<T>(IEnumerable<T> list, int length)
    {
      return GetPermutations(list, length).ToList().ConvertAll(n => n.ToList());
    }


    public static TreeNode<T> GetPermutationTree<T>(T root, IEnumerable<T> list, int length)
    {
      var t = new TreeNode<T>(root);
      var s = new Stack<TreeNode<T>>();
      s.Push(t);
      var p = GetPermutationsList<T>(list, length);

      // Action<TreeNode<T>, object> ConsoleVisitorAction = (TreeNode<T> node, object o) => Console.WriteLine(nodeData.ToString());
      Action<TreeNode<T>, object> ConsoleVisitorAction = (TreeNode<T> node, object o) => // For debugging.
      {
        Console.WriteLine(node.Value.ToString());
      };

      for (int i = 0; i < p.Count; ++i) {
        if (i == 0) { // If first permutation, build first tree branch with each successive node as a child of the previous one.
          for (int j = 0; j < p[i].Count; ++j) {
            var child = s.Peek().AddChild(p[i][j]);
            s.Push(child);
          }
        }
        else { // Integrate other permutations into existing tree.
          // Find starting overlap between current permutation and the one on the stack.
          var sList = s.ToList();
          sList.Reverse();
          sList.RemoveAt(0);
          int j;
          for (j = 0; j < Math.Max(sList.Count, p[i].Count); ++j) {
            if (!EqualityComparer<T>.Default.Equals(sList[j].Value, p[i][j]))
              break;
          }
          // Pop the non-matching nodes off the stack.
          for (int k = 0; k < (sList.Count - j); ++k)
            s.Pop();

          // Add child nodes under node at top of stack.
          for (int k = j; k < p[i].Count; ++k) {
            var child = s.Peek().AddChild(p[i][k]);
            s.Push(child);
          }
        }
      }

      return t;
    }


    // V. http://stackoverflow.com/questions/66893/tree-data-structure-in-c-sharp
    public delegate void TreeVisitor<T>(TreeNode<T> node, object o = null);

    public class TreeNode<T>
    {
      private readonly T _value;
      private readonly List<TreeNode<T>> _children = new List<TreeNode<T>>();

      public TreeNode(T value)
      {
        _value = value;
      }

      public TreeNode<T> this[int i]
      {
        get { return _children[i]; }
      }

      public TreeNode<T> Parent { get; private set; }

      public T Value { get { return _value; } }

      public ReadOnlyCollection<TreeNode<T>> Children
      {
        get { return _children.AsReadOnly(); }
      }

      public TreeNode<T> AddChild(T value)
      {
        var node = new TreeNode<T>(value) { Parent = this };
        _children.Add(node);

        return node;
      }

      public TreeNode<T>[] AddChildren(params T[] values)
      {
        return values.Select(AddChild).ToArray();
      }

      public bool RemoveChild(TreeNode<T> node)
      {
        return _children.Remove(node);
      }

      public bool IsRoot()
      {
        return Parent == null;
      }

      public bool IsLeaf()
      {
        return Children.Count == 0;
      }

      public void Traverse(Action<TreeNode<T>, object> action, object o = null)
      {
        action(this, o);
        foreach (var child in _children)
          child.Traverse(action, o);
      }

      public IEnumerable<T> Flatten()
      {
        return new[] { Value }.Union(_children.SelectMany(x => x.Flatten()));
      }
    }
    #endregion
  }
}
