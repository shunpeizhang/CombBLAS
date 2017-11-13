/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2010-2014, The Regents of the University of California

 Permission is hereby granted, free of charge, to any person obtaining a std::copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, std::copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#ifndef _SP_PAR_VEC_H_
#define _SP_PAR_VEC_H_

#include<iostream>
#include <utility>
#include<vector>
#include "CombBLAS.h"
#include "CommGrid.h"
#include "MPIOp.h"
#include "Operations.h"
#include "SpParMat.h"
#include "promote.h"
namespace combblas {
template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;

/**
  * A sparse vector of length n (with nnz <= n of them being nonzeros) is
  *distributed to
  * diagonal processors in a way that respects ordering of the nonzero indices
  * Example: x = [5,1,6,2,9] for nnz(x)=5 and length(x)=10
  *	we use 4 processors P_00, P_01, P_10, P_11
  * 	Then P_00 owns [1,2] (in the range [0...4]) and P_11 owns rest
  * In the case of A(v,w) type sparse matrix indexing, this doesn't matter
  *because n = nnz
  * 	After all, A(v,w) will have dimensions length(v) x length (w)
  * 	v and w will be of numerical type (NT) "int" and their indices (IT)
  *will be consecutive integers
  * It is possibly that nonzero counts are distributed unevenly
  * Example: x=[1,2,3,4,5] and length(x) = 10, then P_00 would own all the
  *nonzeros and P_11 would hold an empty std::vector
  * Just like in SpParMat case, indices are local to processors (they belong to
  *range [0,...,length-1] on each processor)
  *
  * TODO: Instead of repeated calls to "DiagWorld", this class should be
  *oblivious to the communicator
  * 	  It should just distribute the vector to the MPI::IntraComm that it
  *owns, whether diagonal or whole
 **/

template <class IT, class NT>
class SpParVec {
 public:
  SpParVec();
  SpParVec(IT loclength);
  SpParVec(std::shared_ptr<CommGrid> grid);
  SpParVec(std::shared_ptr<CommGrid> grid, IT loclength);

  //! like operator=, but instead of making a deep std::copy it just steals the
  //! contents.
  //! Useful for places where the "victim" will be distroyed immediately after
  //! the call.
  void stealFrom(SpParVec<IT, NT>& victim);
  SpParVec<IT, NT>& operator+=(const SpParVec<IT, NT>& rhs);
  SpParVec<IT, NT>& operator-=(const SpParVec<IT, NT>& rhs);
  std::ifstream& ReadDistribute(std::ifstream& infile, int master);

  template <typename NNT>
  operator SpParVec<IT, NNT>() const  //!< Type conversion operator
  {
    SpParVec<IT, NNT> CVT(commGrid);
    CVT.ind = std::vector<IT>(ind.begin(), ind.end());
    CVT.num = std::vector<NNT>(num.begin(), num.end());
    CVT.length = length;
  }

  void PrintInfo(std::string vecname) const;
  void iota(IT size, NT first);
  SpParVec<IT, NT> operator()(const SpParVec<IT, IT>& ri)
      const;  //!< SpRef (expects NT of ri to be 0-based)
  void SetElement(IT indx, NT numx);  // element-wise assignment
  NT operator[](IT indx) const;

  // sort the std::vector itself
  // return the permutation std::vector (0-based)
  SpParVec<IT, IT> sort();

  IT getlocnnz() const { return ind.size(); }

  IT getnnz() const {
    IT totnnz = 0;
    IT locnnz = ind.size();
    MPI_Allreduce(&locnnz, &totnnz, 1, MPIType<IT>(), MPI_SUM,
                  commGrid->GetWorld());
    return totnnz;
  }

  IT getTypicalLocLength() const;
  IT getTotalLength(MPI_Comm& comm) const;
  IT getTotalLength() const { return getTotalLength(commGrid->GetWorld()); }

  void setNumToInd() {
    MPI_Comm DiagWorld = commGrid->GetDiagWorld();
    if (DiagWorld != MPI_COMM_NULL)  // Diagonal processors only
    {
      int rank;
      MPI_Comm_rank(DiagWorld, &rank);

      IT n_perproc = getTypicalLocLength();
      IT offset = static_cast<IT>(rank) * n_perproc;

      std::transform(ind.begin(), ind.end(), num.begin(),
                bind2nd(std::plus<IT>(), offset));
    }
  }

  template <typename _UnaryOperation>
  void Apply(_UnaryOperation __unary_op) {
    std::transform(num.begin(), num.end(), num.begin(), __unary_op);
  }

  template <typename _UnaryOperation>
  void ApplyInd(_UnaryOperation __unary_op) {
    MPI_Comm DiagWorld = commGrid->GetDiagWorld();
    if (DiagWorld != MPI_COMM_NULL)  // Diagonal processors only
    {
      int rank;
      MPI_Comm_rank(DiagWorld, &rank);

      IT n_perproc = getTypicalLocLength();
      IT offset = static_cast<IT>(rank) * n_perproc;

      for (IT i = 0; i < ind.size(); i++) {
        num[i] = __unary_op(ind[i] + offset, num[i]);
      }
    }
  }

  int offset() {
    MPI_Comm DiagWorld = commGrid->GetDiagWorld();
    if (DiagWorld != MPI_COMM_NULL)  // Diagonal processors only
    {
      int rank;
      MPI_Comm_rank(DiagWorld, &rank);

      IT n_perproc = getTypicalLocLength();
      IT offset = static_cast<IT>(rank) * n_perproc;
      return offset;
    } else {
      return -1;
    }
  }

  //! Filters elements using op(index, nonzero)
  template <typename _Operation>
  void Filter(_Operation op) {
    MPI_Comm DiagWorld = commGrid->GetDiagWorld();
    if (DiagWorld != MPI_COMM_NULL)  // Diagonal processors only
    {
      int rank;
      MPI_Comm_rank(DiagWorld, &rank);

      IT n_perproc = getTypicalLocLength();
      IT offset = static_cast<IT>(rank) * n_perproc;

      std::vector<IT> new_ind;
      std::vector<NT> new_num;

      for (IT i = 0; i < ind.size(); i++) {
        if (op(ind[i] + offset, num[i])) {
          new_ind.push_back(ind[i]);
          new_num.push_back(num[i]);
        }
      }

      ind = new_ind;
      num = new_num;
    }
  }

  template <typename _BinaryOperation>
  NT Reduce(_BinaryOperation __binary_op, NT init);

  void DebugPrint();
  std::shared_ptr<CommGrid> getcommgrid() const { return commGrid; }
  NT NOT_FOUND;
  std::vector<IT>& getind() { return ind; }
  std::vector<NT>& getnum() { return num; }
  IT& getlength() { return length; }
  const std::vector<IT>& getind() const { return ind; }
  const std::vector<NT>& getnum() const { return num; }

 private:
  std::shared_ptr<CommGrid> commGrid;
  std::vector<IT> ind;  // ind.size() give the number of nonzeros
  std::vector<NT> num;
  IT length;  // actual local length of the std::vector (including zeros)
  bool diagonal;

  template <class IU, class NU>
  friend class SpParVec;

  template <class IU, class NU>
  friend class DenseParVec;

  template <class IU, class NU, class UDER>
  friend class SpParMat;

  template <typename SR, typename IU, typename NUM, typename NUV, typename UDER>
  friend SpParVec<IU, typename SR::T_promote> SpMV(
      const SpParMat<IU, NUM, UDER>& A, const SpParVec<IU, NUV>& x);

  template <typename IU, typename NU1, typename NU2>
  friend SpParVec<IU, typename promote_trait<NU1, NU2>::T_promote> EWiseMult(
      const SpParVec<IU, NU1>& V, const DenseParVec<IU, NU2>& W, bool exclude,
      NU2 zero);

  template <typename IU>
  friend void RandPerm(SpParVec<IU, IU>& V);  // called on an existing object,
                                              // randomly permutes it

  template <typename IU>
  friend void RenameVertices(DistEdgeList<IU>& DEL);
};
}

#include "SpParVec.cpp"
#endif
